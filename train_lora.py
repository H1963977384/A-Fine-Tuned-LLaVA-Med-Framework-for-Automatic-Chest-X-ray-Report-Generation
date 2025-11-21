import argparse
import os
import json
import math
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

from PIL import Image
from transformers import set_seed, logging

# HuggingFace / LLaVA model imports for medical vision-language tasks
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, process_images

# Parameter-Efficient Fine-Tuning (PEFT) for efficient adaptation
from peft import LoraConfig, get_peft_model

# Configure logging and random seeds for reproducibility
logging.set_verbosity_error()
set_seed(0)
disable_torch_init()

# Project directory structure configuration
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON_PATH = PROJECT_ROOT / "data" / "train_report.json"
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "images" / "images_normalized"


# -------------------------
# Dataset Implementation
# -------------------------
class LLaVAMedDataset(Dataset):
    """
    A PyTorch Dataset class for loading and preprocessing medical report data 
    for LLaVA-Med model fine-tuning.
    
    Each data sample contains an image and conversation pairs (question-answer).
    The dataset constructs proper prompt formatting with image tokens and 
    handles tokenization with appropriate loss masking.
    
    Args:
        reports: List of report dictionaries containing image paths and conversations
        tokenizer: Pre-trained tokenizer for text processing
        image_processor: Image processor for vision input preprocessing
        model_config: Configuration object from the loaded model
        image_dir: Root directory containing medical images
        context_len: Maximum sequence length for tokenized inputs
    """

    def __init__(self, reports, tokenizer, image_processor, model_config, image_dir, context_len=2048):
        self.reports = reports
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.image_dir = image_dir
        self.context_len = context_len
        # Extract model-specific configuration for image token handling
        self.mm_use_im_start_end = getattr(model_config, "mm_use_im_start_end", False)
        self.DEFAULT_IM_START_TOKEN = DEFAULT_IM_START_TOKEN
        self.DEFAULT_IM_END_TOKEN = DEFAULT_IM_END_TOKEN
        self.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        # Use Vicuna v1 conversation template for consistent prompt formatting
        self.conv_template = conv_templates["vicuna_v1"]

    def build_prompt_and_target(self, report):
        """
        Constructs the training prompt and target answer from a report sample.
        
        The method formats the conversation according to the template, inserts
        image tokens appropriately, and prepares the ground truth response.
        
        Args:
            report: Dictionary containing 'conversations' with question-answer pairs
            
        Returns:
            tuple: (formatted_prompt, target_answer) strings
        """
        # Extract and clean the question text
        question = report['conversations'][0]['value']
        qs = question.replace(DEFAULT_IMAGE_TOKEN, '').strip()

        # Format question with image tokens based on model configuration
        if self.mm_use_im_start_end:
            qs_for_prompt = self.DEFAULT_IM_START_TOKEN + self.DEFAULT_IMAGE_TOKEN + self.DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs_for_prompt = self.DEFAULT_IMAGE_TOKEN + '\n' + qs

        # Build conversation using template
        conv = self.conv_template.copy()
        conv.append_message(conv.roles[0], qs_for_prompt)
        conv.append_message(conv.roles[1], None)  # Placeholder for model response
        prompt = conv.get_prompt()

        # Extract ground truth answer from conversation
        target = report['conversations'][1]['value'].strip()
        return prompt, target

    def encode_pair(self, prompt, target):
        """
        Tokenizes prompt and target, applies loss masking, and handles sequence length constraints.
        
        The prompt portion is masked with -100 in labels to exclude from loss calculation,
        while the target portion is included for training.
        
        Args:
            prompt: Input prompt string with conversation history
            target: Target answer string for training
            
        Returns:
            tuple: (input_ids tensor, labels tensor) with proper padding and truncation
        """
        # Tokenize prompt and target separately for precise loss masking
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = self.tokenizer(target, add_special_tokens=False)["input_ids"]

        # Concatenate prompt, target, and EOS token
        input_ids = prompt_ids + target_ids
        if self.tokenizer.eos_token_id is not None:
            input_ids = input_ids + [self.tokenizer.eos_token_id]

        # Handle sequence length constraints with priority on preserving target tokens
        if len(input_ids) > self.context_len:
            # Calculate available space for prompt while preserving target
            keep_prompt_len = max(0, self.context_len - len(target_ids) - (1 if self.tokenizer.eos_token_id is not None else 0))
            prompt_ids = prompt_ids[-keep_prompt_len:]
            input_ids = prompt_ids + target_ids
            if self.tokenizer.eos_token_id is not None:
                input_ids = input_ids + [self.tokenizer.eos_token_id]

        # Create labels: mask prompt with -100, keep target for loss calculation
        labels = [-100] * len(prompt_ids) + target_ids
        if self.tokenizer.eos_token_id is not None:
            labels = labels + [self.tokenizer.eos_token_id]

        # Apply padding to reach context length
        pad_len = self.context_len - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
            labels = labels + [-100] * pad_len
        else:
            # Truncate if still exceeding context length (should not happen with above logic)
            input_ids = input_ids[:self.context_len]
            labels = labels[:self.context_len]

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

    def load_image_tensor(self, image_name):
        """
        Loads and preprocesses an image from disk into model-ready tensor format.
        
        Args:
            image_name: Filename of the image to load
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        image_tensor = process_images([image], self.image_processor, None)[0]
        return image_tensor

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.reports)

    def __getitem__(self, idx):
        """
        Retrieves and processes a single dataset sample.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            dict: Contains tokenized input_ids, labels, and processed image tensor
        """
        report = self.reports[idx]
        prompt, target = self.build_prompt_and_target(report)
        input_ids, labels = self.encode_pair(prompt, target)
        image_tensor = self.load_image_tensor(report['image'])
        return {
            "input_ids": input_ids,
            "labels": labels,
            "images": image_tensor
        }


# -------------------------
# Training Utilities
# -------------------------
def collate_fn(batch):
    """
    Custom collate function for DataLoader to stack individual samples into batches.
    
    Args:
        batch: List of samples from dataset __getitem__
        
    Returns:
        dict: Batched tensors for input_ids, labels, and images
    """
    input_ids = torch.stack([item["input_ids"] for item in batch], dim=0)
    labels = torch.stack([item["labels"] for item in batch], dim=0)
    images = torch.stack([item["images"] for item in batch], dim=0)
    return {"input_ids": input_ids, "labels": labels, "images": images}


def train(args):
    """
    Main training loop for fine-tuning LLaVA-Med model using LoRA.
    
    Loads the base model, applies LoRA adaptation, sets up data loading,
    and runs the training process with mixed precision and gradient accumulation.
    
    Args:
        args: Command line arguments containing all training configuration
    """
    # Load pre-trained model and components
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=args.model_path,
        model_base=None,  # Load base model directly (not from separate base)
        model_name=model_name,
        load_8bit=False,
        load_4bit=False,
        device_map=None,  # Manual device placement
        device="cuda"
    )
    model.config.max_sequence_length = context_len

    # Ensure tokenizer has required special tokens
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "</s>"})

    # Configure and apply LoRA for parameter-efficient fine-tuning
    lora_config = LoraConfig(
        r=args.lora_r,  # LoRA rank
        lora_alpha=args.lora_alpha,  # LoRA alpha parameter
        target_modules=args.target_modules.split(","),  # Modules to apply LoRA to
        lora_dropout=args.lora_dropout,  # Dropout for LoRA layers
        bias="none",  # No bias treatment
        task_type="CAUSAL_LM"  # Causal language modeling task
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Display parameter efficiency

    # Set device and move model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load and prepare training data
    with open(args.json_file, "r") as f:
        reports = [json.loads(line) for line in f]

    dataset = LLaVAMedDataset(reports, tokenizer, image_processor, model.config, args.image_dir, context_len=context_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # Initialize optimizer and gradient scaler for mixed precision
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler()

    # Training state tracking
    global_step = 0
    model.train()
    
    # Epoch training loop
    for epoch in range(args.epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        running_loss = 0.0
        
        # Batch training loop
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            images = batch["images"].to(device)

            # Mixed precision forward pass
            with autocast():
                outputs = model(input_ids=input_ids, labels=labels, images=images)
                loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
                loss = loss / args.gradient_accumulation_steps  # Normalize for accumulation

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()

            # Gradient accumulation and optimizer step
            if (step + 1) % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1
                pbar.set_postfix({"loss": loss.item() * args.gradient_accumulation_steps})

            running_loss += loss.item() * args.gradient_accumulation_steps

            # Intermediate checkpoint saving
            if global_step > 0 and global_step % args.save_steps == 0:
                save_dir = os.path.join(args.output_dir, f"checkpoint-step-{global_step}")
                os.makedirs(save_dir, exist_ok=True)
                print(f"\nSaving LoRA adapters to {save_dir}")
                model.save_pretrained(save_dir)

        # Epoch statistics and checkpointing
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1} finished. avg loss: {epoch_loss:.4f}")

        save_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch+1}")
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        print(f"Saved epoch checkpoint to {save_dir}")

    # Final model saving
    final_save = os.path.join(args.output_dir, "lora_final")
    os.makedirs(final_save, exist_ok=True)
    model.save_pretrained(final_save)
    print(f"Final LoRA saved to {final_save}")


def run_model_inference(tokenizer, model, image_processor, report, device="cuda"):
    """
    Run inference on a single report using the fine-tuned model.
    
    Processes the input question, formats with image tokens, and generates
    model response using beam search or sampling.
    
    Args:
        tokenizer: Pre-trained tokenizer for text processing
        model: Fine-tuned model for inference
        image_processor: Image preprocessing module
        report: Dictionary containing image path and conversation
        device: Device to run inference on
        
    Returns:
        str: Generated model response
    """
    question = report['conversations'][0]['value']
    qs = question.replace(DEFAULT_IMAGE_TOKEN, '').strip()
    
    # Format question with image tokens
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    # Build conversation prompt
    conv = conv_templates["vicuna_v1"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Tokenize input
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    # Load and process image
    image_path = os.path.join(args.image_dir, report['image'])
    image = Image.open(image_path)
    image_tensor = process_images([image], image_processor, model.config)[0]

    # Generate response
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().cuda(),
            do_sample=True,
            temperature=0.2,
            top_p=None,
            num_beams=1,
            max_new_tokens=1024,
            use_cache=True)
        
    output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return output


# -------------------------
# Command Line Interface
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune LLaVA-Med model using LoRA for medical report generation")
    
    # Model and data configuration
    parser.add_argument("--model_path", type=str, default="microsoft/llava-med-v1.5-mistral-7b",
                        help="Base LLaVA-Med model path (HuggingFace model ID or local directory)")
    parser.add_argument("--json_file", type=str, default=str(DEFAULT_JSON_PATH), 
                        help="JSON annotation file with one report entry per line")
    parser.add_argument("--image_dir", type=str, default=str(DEFAULT_IMAGE_DIR),
                        help="Directory containing medical images")
    parser.add_argument("--output_dir", type=str, default="./lora_output", 
                        help="Output directory for storing LoRA adapters and checkpoints")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, 
                        help="Number of steps for gradient accumulation")
    parser.add_argument("--save_steps", type=int, default=200, 
                        help="Save checkpoint every X optimization steps")
    
    # LoRA configuration
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate")
    parser.add_argument("--target_modules", type=str, 
                        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                        help="Comma-separated list of transformer modules to apply LoRA to")
    
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Start training
    train(args)

