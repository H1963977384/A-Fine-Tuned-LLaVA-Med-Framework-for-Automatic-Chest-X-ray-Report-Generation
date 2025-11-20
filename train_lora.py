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

# HF / LLAVA imports (保持你的工程路径)
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, process_images

# PEFT
from peft import LoraConfig, get_peft_model

logging.set_verbosity_error()
set_seed(0)
disable_torch_init()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON_PATH = PROJECT_ROOT / "data" / "train_report.json"
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "IU_XRay_Cleaned_Dataset" / "images"

# -------------------------
# Dataset
# -------------------------
class LLaVAMedDataset(Dataset):
    """
    每个 item 是一个 report dict，包含 'image' 和 'conversations' (list, first elem 是 question, second elem 是 answer)
    按照 run_model 的方式构造 prompt（含图像 token），并把 prompt + answer 拼接。
    labels: prompt 部分为 -100（不计算 loss），answer 部分为 token ids。
    """

    def __init__(self, reports, tokenizer, image_processor, model_config, image_dir, context_len=2048):
        self.reports = reports
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.image_dir = image_dir
        self.context_len = context_len
        self.mm_use_im_start_end = getattr(model_config, "mm_use_im_start_end", False)
        self.DEFAULT_IM_START_TOKEN = DEFAULT_IM_START_TOKEN
        self.DEFAULT_IM_END_TOKEN = DEFAULT_IM_END_TOKEN
        self.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self.conv_template = conv_templates["vicuna_v1"]

    def build_prompt_and_target(self, report):
        # question text
        question = report['conversations'][0]['value']
        qs = question.replace(DEFAULT_IMAGE_TOKEN, '').strip()

        if self.mm_use_im_start_end:
            qs_for_prompt = self.DEFAULT_IM_START_TOKEN + self.DEFAULT_IMAGE_TOKEN + self.DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs_for_prompt = self.DEFAULT_IMAGE_TOKEN + '\n' + qs

        # conversation template
        conv = self.conv_template.copy()
        conv.append_message(conv.roles[0], qs_for_prompt)
        conv.append_message(conv.roles[1], None)  # gpt reply placeholder
        prompt = conv.get_prompt()

        # target is the ground-truth answer (second conversation element)
        target = report['conversations'][1]['value'].strip()
        # Sometimes target may include special tokens or newlines; fine.
        return prompt, target

    def encode_pair(self, prompt, target):
        # encode prompt and target separately so we can mask prompt labels
        # NOTE: don't add special tokens twice; use add_special_tokens=False for concatenation control.
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = self.tokenizer(target, add_special_tokens=False)["input_ids"]

        # concatenate and ensure within context length
        input_ids = prompt_ids + target_ids + [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id is not None else prompt_ids + target_ids

        if len(input_ids) > self.context_len:
            # 如果过长，优先截断 prompt（保留尽可能多的 target）
            # strategy: keep last `context_len - len(target_ids)` tokens of prompt
            keep_prompt_len = max(0, self.context_len - len(target_ids) - (1 if self.tokenizer.eos_token_id is not None else 0))
            prompt_ids = prompt_ids[-keep_prompt_len:]
            input_ids = prompt_ids + target_ids
            if self.tokenizer.eos_token_id is not None:
                input_ids = input_ids + [self.tokenizer.eos_token_id]

        labels = [-100] * len(prompt_ids) + target_ids
        if self.tokenizer.eos_token_id is not None:
            labels = labels + [self.tokenizer.eos_token_id]

        # pad to context_len
        pad_len = self.context_len - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
            labels = labels + [-100] * pad_len
        else:
            # already handled clipping above, but be safe:
            input_ids = input_ids[:self.context_len]
            labels = labels[:self.context_len]

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

    def load_image_tensor(self, image_name):
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        image_tensor = process_images([image], self.image_processor, None)[0]
        return image_tensor

    def __len__(self):
        return len(self.reports)

    def __getitem__(self, idx):
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
# Training / Utility
# -------------------------
def collate_fn(batch):
    # batch is list of dicts; stack tensors
    input_ids = torch.stack([item["input_ids"] for item in batch], dim=0)
    labels = torch.stack([item["labels"] for item in batch], dim=0)
    # images: each is [C,H,W]; stack -> [B,C,H,W]
    images = torch.stack([item["images"] for item in batch], dim=0)
    return {"input_ids": input_ids, "labels": labels, "images": images}

def train(args):
    # load base model (no LoRA yet)
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=args.model_path,
        model_base=None,  # we're loading the base model first (for training)
        model_name=model_name,
        load_8bit=False,
        load_4bit=False,
        device_map=None,  # we'll move manually
        device="cuda"
    )
    model.config.max_sequence_length = context_len

    # ensure tokenizer has pad/eos tokens
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "</s>"})

    # inject LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules.split(","),
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # dataset
    with open(args.json_file, "r") as f:
        reports = [json.loads(line) for line in f]

    dataset = LLaVAMedDataset(reports, tokenizer, image_processor, model.config, args.image_dir, context_len=context_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # optimizer & scaler
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler()

    global_step = 0
    model.train()
    for epoch in range(args.epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        running_loss = 0.0
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            images = batch["images"].to(device)

            with autocast():
                outputs = model(input_ids=input_ids, labels=labels, images=images)
                # many LM wrappers return loss in outputs.loss
                loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
                loss = loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1
                pbar.set_postfix({"loss": loss.item()*args.gradient_accumulation_steps})

            running_loss += loss.item() * args.gradient_accumulation_steps

            # optional save
            if global_step > 0 and global_step % args.save_steps == 0:
                save_dir = os.path.join(args.output_dir, f"checkpoint-step-{global_step}")
                os.makedirs(save_dir, exist_ok=True)
                print(f"\nSaving LoRA adapters to {save_dir}")
                model.save_pretrained(save_dir)

        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1} finished. avg loss: {epoch_loss:.4f}")

        # epoch-end save
        save_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch+1}")
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        print(f"Saved epoch checkpoint to {save_dir}")

    # final save
    final_save = os.path.join(args.output_dir, "lora_final")
    os.makedirs(final_save, exist_ok=True)
    model.save_pretrained(final_save)
    print(f"Final LoRA saved to {final_save}")


# keep your original inference function adapted to new tokenizer/model objects
def run_model_inference(tokenizer, model, image_processor, report, device="cuda"):
    question = report['conversations'][0]['value']
    qs = question.replace(DEFAULT_IMAGE_TOKEN, '').strip()
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv = conv_templates["vicuna_v1"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    image_path = os.path.join(args.image_dir, report['image'])
    image = Image.open(image_path)
    image_tensor = process_images([image], image_processor, model.config)[0]

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
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="microsoft/llava-med-v1.5-mistral-7b",
                        help="base LLaVA-Med model path (HuggingFace id or local)")
    parser.add_argument("--json_file", type=str, default=str(DEFAULT_JSON_PATH), help="每行一个 json 的标注文件")
    parser.add_argument("--image_dir", type=str, default=str(DEFAULT_IMAGE_DIR),
                        help="图片目录")
    parser.add_argument("--output_dir", type=str, default="./lora_output", help="保存 LoRA 的目录")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                        help="逗号分隔的 target module 列表")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    train(args)