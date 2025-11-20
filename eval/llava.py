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

warnings.filterwarnings('ignore')
logging.set_verbosity_error()

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

set_seed(0)
disable_torch_init()
sys.path.append('../LLaVA-Med')
model_path = 'microsoft/llava-med-v1.5-mistral-7b'

model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=model_name,
    load_8bit=False,
    load_4bit=False,
    device_map='auto',
    device='cuda'
)

def run_model(report):
    question = report['conversations'][0]['value']+" The Findings and Impression sections must be logically consistent. A finding cannot both be present and absent. State each finding only once. Do not create looping or repetitive text."
    questions = get_chunk(question, 1, 0)
    qs = questions.replace(DEFAULT_IMAGE_TOKEN, '').strip()
	
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
    
    conv = conv_templates["vicuna_v1"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    image_path = './data/images/'+ report['image']
    image = Image.open(image_path)
    image_tensor = process_images([image], image_processor, model.config)[0]
    
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    
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

def calculate_precision_recall(reference, candidate):
	ref_words = set(' '.join(reference).lower().split())
	cand_words = set(candidate.lower().split())
	common_words = ref_words.intersection(cand_words)
	precision = len(common_words) / len(cand_words) if len(cand_words) > 0 else 0
	recall = len(common_words) / len(ref_words) if len(ref_words) > 0 else 0
	f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
	return precision, recall, f1


def calculate_sacrebleu_all(references, candidate):
    smoothie = SmoothingFunction().method7
    
    bleu1 = sentence_bleu(references, candidate, weights=(1, 0, 0, 0), smoothing_function=smoothie)
    bleu2 = sentence_bleu(references, candidate, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
    bleu3 = sentence_bleu(references, candidate, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie)
    bleu4 = sentence_bleu(references, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
    
    return bleu1, bleu2, bleu3, bleu4

parser = argparse.ArgumentParser(description='input testing file')
parser.add_argument('--json_file', type=str, required=True, help='testing file')
args = parser.parse_args()
json_file = args.json_file
with open(json_file, 'r') as f:
    reports = [json.loads(line) for line in f]

count = 0;
finding_results = {'B-1': 0, 'B-2': 0, 'B-3': 0, 'B-4': 0, 'METEOR': 0, 'ROUGE-1': 0, 'ROUGE-2': 0, 'ROUGE-L': 0, 'F1': 0, 'Precision': 0, 'Recall': 0}
impression_results = {'B-1': 0, 'B-2': 0, 'B-3': 0, 'B-4': 0, 'METEOR': 0, 'ROUGE-1': 0, 'ROUGE-2': 0, 'ROUGE-L': 0, 'F1': 0, 'Precision': 0, 'Recall': 0}
rouge = Rouge()
for report in tqdm(reports):
    count += 1
    ori = report["conversations"][1]["value"]
    output = run_model(report)
    
    ori_tokens = [ori.split()]
    pre_tokens = output.split()

    IB1, IB2, IB3, IB4 = calculate_sacrebleu_all(ori_tokens, pre_tokens)
    finding_results['B-1'] += IB1
    finding_results['B-2'] += IB2
    finding_results['B-3'] += IB3
    finding_results['B-4'] += IB4

    finding_results['METEOR'] += meteor_score(ori_tokens, pre_tokens)

    finding_rouge = rouge.get_scores([ori], [output], avg=True)
    finding_results['ROUGE-1'] += finding_rouge['rouge-1']['f']
    finding_results['ROUGE-2'] += finding_rouge['rouge-2']['f']
    finding_results['ROUGE-L'] += finding_rouge['rouge-l']['f']

    finding_precision, finding_recall, finding_f1 = calculate_precision_recall([ori], output)
    finding_results['Precision'] += finding_precision
    finding_results['Recall'] += finding_recall
    finding_results['F1'] += finding_f1

    print(f"【Finding】    B-1: {finding_results['B-1']/count:.2f}, B-2: {finding_results['B-2']/count:.2f}, B-3: {finding_results['B-3']/count:.2f}, B-4: {finding_results['B-4']/count:.2f}, METEOR: {finding_results['METEOR']/count:.2f}, ROUGE-1: {finding_results['ROUGE-1']/count:.2f}, ROUGE-2: {finding_results['ROUGE-2']/count:.2f}, ROUGE-L: {finding_results['ROUGE-L']/count:.2f}, Precision: {finding_results['Precision']/count:.2f}, Recall: {finding_results['Recall']/count:.2f}, F1: {finding_results['F1']/count:.2f}")