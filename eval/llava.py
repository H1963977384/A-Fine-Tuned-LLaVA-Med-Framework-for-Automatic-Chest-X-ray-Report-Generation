# Model evaluation script for LLaVA-Med fine-tuned model performance assessment
# Computes multiple NLP metrics (BLEU, ROUGE, METEOR, F1) on radiology report generation

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from PIL import Image
from rouge import Rouge
from tqdm import tqdm
from transformers import set_seed, logging

import argparse
import jieba
import json
import math
import re
import sys
import torch
import warnings

# Suppress warnings and reduce logging verbosity for cleaner output
warnings.filterwarnings('ignore')
logging.set_verbosity_error()

def split_list(lst, n):
    """
    Divide a list into approximately equal-sized chunks for distributed processing.
    
    Args:
        lst (list): Input list to be partitioned
        n (int): Number of chunks to create
        
    Returns:
        list: List of n sublists with roughly equal sizes
    """
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    """
    Extract a specific chunk from a list that has been divided into n parts.
    
    Args:
        lst (list): Original list to chunk
        n (int): Total number of chunks
        k (int): Index of chunk to retrieve (0-indexed)
        
    Returns:
        list: The k-th chunk from the partitioned list
    """
    chunks = split_list(lst, n)
    return chunks[k]

# Initialize model environment with fixed random seed for reproducibility
set_seed(0)
disable_torch_init()
sys.path.append('../LLaVA-Med')

# Load pre-trained LLaVA-Med model for inference
model_path = 'microsoft/llava-med-v1.5-mistral-7b'
model_name = get_model_name_from_path(model_path)

# Initialize model components: tokenizer, vision encoder, and language model
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,  # Load base model without fine-tuned weights
    model_name=model_name,
    load_8bit=False,   # Full precision inference
    load_4bit=False,   # Full precision inference  
    device_map='auto', # Automatic GPU memory management
    device='cuda'      # Use GPU acceleration
)

def run_model(report):
    """
    Generate radiology report for a given medical image using the LLaVA-Med model.
    
    Processes the input prompt with image tokens, handles conversation formatting,
    and generates text output with controlled sampling parameters.
    
    Args:
        report (dict): Medical report data containing image path and conversation
        
    Returns:
        str: Generated radiology report text from the model
    """
    # Construct enhanced prompt with consistency constraints for medical reporting
    question = report['conversations'][0]['value']+" The Findings and Impression sections must be logically consistent. A finding cannot both be present and absent. State each finding only once. Do not create looping or repetitive text."
    questions = get_chunk(question, 1, 0)
    qs = questions.replace(DEFAULT_IMAGE_TOKEN, '').strip()
    
    # Format question with appropriate image tokens based on model configuration
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
    
    # Build conversation using Vicuna template for consistent formatting
    conv = conv_templates["vicuna_v1"].copy()
    conv.append_message(conv.roles[0], qs)  # User question
    conv.append_message(conv.roles[1], None)  # Assistant response placeholder
    prompt = conv.get_prompt()
    
    # Tokenize input text with special handling for image tokens
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    # Load and preprocess medical image for model input
    image_path = './data/images/'+ report['image']
    image = Image.open(image_path)
    image_tensor = process_images([image], image_processor, model.config)[0]
    
    # Configure stopping criteria to terminate generation at conversation boundaries
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    
    # Generate text with controlled sampling parameters for consistent output
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().cuda(),  # Half precision for memory efficiency
            do_sample=True,      # Enable probabilistic sampling
            temperature=0.2,     # Low temperature for focused sampling
            top_p=None,          # No nucleus sampling
            num_beams=1,         # Greedy decoding (no beam search)
            max_new_tokens=1024, # Maximum generation length
            use_cache=True)      # Enable KV caching for speed
    
    # Decode token IDs to text and clean special tokens
    output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return output

def calculate_precision_recall(reference, candidate):
    """
    Calculate word-level precision, recall, and F1 score between reference and candidate texts.
    
    Args:
        reference (str): Ground truth text
        candidate (str): Model-generated text
        
    Returns:
        tuple: (precision, recall, f1) scores as floats
    """
    ref_words = set(' '.join(reference).lower().split())
    cand_words = set(candidate.lower().split())
    common_words = ref_words.intersection(cand_words)
    
    precision = len(common_words) / len(cand_words) if len(cand_words) > 0 else 0
    recall = len(common_words) / len(ref_words) if len(ref_words) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

def calculate_sacrebleu_all(references, candidate):
    """
    Compute BLEU scores at multiple n-gram levels for text generation evaluation.
    
    BLEU (Bilingual Evaluation Understudy) measures n-gram overlap between
    generated text and reference texts, with smoothing for short sequences.
    
    Args:
        references (list): List of reference text tokens
        candidate (str): Candidate text to evaluate
        
    Returns:
        tuple: BLEU scores for 1-gram, 2-gram, 3-gram, and 4-gram matches
    """
    smoothie = SmoothingFunction().method7
    
    bleu1 = sentence_bleu(references, candidate, weights=(1, 0, 0, 0), smoothing_function=smoothie)
    bleu2 = sentence_bleu(references, candidate, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
    bleu3 = sentence_bleu(references, candidate, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie)
    bleu4 = sentence_bleu(references, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
    
    return bleu1, bleu2, bleu3, bleu4

# Configure command-line interface for evaluation script
parser = argparse.ArgumentParser(description='input testing file')
parser.add_argument('--json_file', type=str, required=True, help='Path to JSON file containing test cases with image references and ground truth reports')
args = parser.parse_args()
json_file = args.json_file

# Load test dataset containing medical reports for evaluation
with open(json_file, 'r') as f:
    reports = [json.loads(line) for line in f]

# Initialize accumulators for evaluation metrics
count = 0
# Store cumulative scores for findings section evaluation
finding_results = {'B-1': 0, 'B-2': 0, 'B-3': 0, 'B-4': 0, 'METEOR': 0, 'ROUGE-1': 0, 'ROUGE-2': 0, 'ROUGE-L': 0, 'F1': 0, 'Precision': 0, 'Recall': 0}
# Store cumulative scores for impression section evaluation  
impression_results = {'B-1': 0, 'B-2': 0, 'B-3': 0, 'B-4': 0, 'METEOR': 0, 'ROUGE-1': 0, 'ROUGE-2': 0, 'ROUGE-L': 0, 'F1': 0, 'Precision': 0, 'Recall': 0}

# Initialize ROUGE scorer for text similarity metrics
rouge = Rouge()

# Evaluate model performance on each test case
for report in tqdm(reports):
    count += 1
    ori = report["conversations"][1]["value"]  # Ground truth report
    output = run_model(report)  # Model-generated report
    
    # Prepare tokenized versions for metric computation
    ori_tokens = [ori.split()]
    pre_tokens = output.split()

    # Calculate BLEU scores at different n-gram levels
    IB1, IB2, IB3, IB4 = calculate_sacrebleu_all(ori_tokens, pre_tokens)
    finding_results['B-1'] += IB1
    finding_results['B-2'] += IB2
    finding_results['B-3'] += IB3
    finding_results['B-4'] += IB4

    # Calculate METEOR score for semantic similarity
    finding_results['METEOR'] += meteor_score(ori_tokens, pre_tokens)

    # Calculate ROUGE scores for n-gram overlap and longest common subsequence
    finding_rouge = rouge.get_scores([ori], [output], avg=True)
    finding_results['ROUGE-1'] += finding_rouge['rouge-1']['f']
    finding_results['ROUGE-2'] += finding_rouge['rouge-2']['f']
    finding_results['ROUGE-L'] += finding_rouge['rouge-l']['f']

    # Calculate precision, recall, and F1 for word-level matching
    finding_precision, finding_recall, finding_f1 = calculate_precision_recall([ori], output)
    finding_results['Precision'] += finding_precision
    finding_results['Recall'] += finding_recall
    finding_results['F1'] += finding_f1

    # Print running averages for real-time monitoring
    print(f"【Finding】    B-1: {finding_results['B-1']/count:.2f}, B-2: {finding_results['B-2']/count:.2f}, B-3: {finding_results['B-3']/count:.2f}, B-4: {finding_results['B-4']/count:.2f}, METEOR: {finding_results['METEOR']/count:.2f}, ROUGE-1: {finding_results['ROUGE-1']/count:.2f}, ROUGE-2: {finding_results['ROUGE-2']/count:.2f}, ROUGE-L: {finding_results['ROUGE-L']/count:.2f}, Precision: {finding_results['Precision']/count:.2f}, Recall: {finding_results['Recall']/count:.2f}, F1: {finding_results['F1']/count:.2f}")
