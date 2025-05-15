import json
import torch
from tqdm import tqdm
from transformers import XLMRobertaTokenizerFast, XLMRobertaForSequenceClassification
import os

# load model and tokenizer
tokenizer = XLMRobertaTokenizerFast.from_pretrained(
    's-nlp/xlmr_formality_classifier')
model = XLMRobertaForSequenceClassification.from_pretrained(
    's-nlp/xlmr_formality_classifier')

id2formality = {0: "formal", 1: "informal"}

# input files
input_files = [
    '/home/wendell/root/LM-Steer-main/LM-Steer-main/logs/formality/gpt2-large-process-formal-1-pos-predictions-500.jsonl',
    '/home/wendell/root/LM-Steer-main/LM-Steer-main/logs/formality-2/gpt2-process-formal-2-pos-predictions-500.jsonl',
    '/home/wendell/root/LM-Steer-main/LM-Steer-main/logs/formality-2/gpt2-process-formal-2-neg-predictions-500.jsonl',
    '/home/wendell/root/LM-Steer-main/LM-Steer-main/logs/formality-2/gpt2-medium-process-formal-2-neg-predictions-500.jsonl',
    '/home/wendell/root/LM-Steer-main/LM-Steer-main/logs/formality-2/gpt2-medium-process-formal-2-pos-predictions-500.jsonl',
    '/home/wendell/root/LM-Steer-main/LM-Steer-main/logs/formality-2/gpt2-large-process-formal-2-pos-predictions-500.jsonl',
    '/home/wendell/root/LM-Steer-main/LM-Steer-main/logs/formality-2/gpt2-large-process-formal-2-neg-predictions-500.jsonl',
    '/home/wendell/root/LM-Steer-main/LM-Steer-main/logs/origin/gpt2-process-origin-predictions-500.jsonl',
    '/home/wendell/root/LM-Steer-main/LM-Steer-main/logs/origin/gpt2-medium-process-origin-predictions-500.jsonl',
    '/home/wendell/root/LM-Steer-main/LM-Steer-main/logs/origin/gpt2-large-process-origin-predictions-500.jsonl',
]

# check if all input files exist
all_files_exist = True
missing_files = []

for file_path in input_files:
    if not os.path.exists(file_path):
        all_files_exist = False
        missing_files.append(file_path)
        print(f"The file doesn't exist: {file_path}")
    else:
        print(f"The file exists: {file_path}")


# process each input file
for input_file in input_files:
    print(f"\nProcessing: {input_file}")

    # 1. read the input file
    texts = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            texts.append(json.loads(line)['text'])

    if not texts:
        print(f"{input_file} is empty.")
        continue

    # initialize a list to store all formality scores
    all_formality_scores = []

    # 2. process the texts in batches
    batch_size = 8
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for i in tqdm(range(total_batches), desc="Progress"):
        batch_texts = texts[i*batch_size: (i+1)*batch_size]

        encoding = tokenizer(
            batch_texts,
            add_special_tokens=True,
            return_token_type_ids=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        with torch.no_grad():
            output = model(**encoding)

        batch_formality_scores = [
            {id2formality[idx]: score for idx,
                score in enumerate(text_scores.tolist())}
            for text_scores in output.logits.softmax(dim=1)
        ]

        all_formality_scores.extend(batch_formality_scores)

    # 3. generate the output filename
    directory, filename = os.path.split(input_file)
    output_filename = os.path.join(directory, f"eval-{filename}")

    # 4. save the results to a new file
    with open(output_filename, 'w', encoding='utf-8') as f:
        for score_dict in all_formality_scores:
            f.write(json.dumps(score_dict, ensure_ascii=False) + '\n')

        # alculate the average formality scores
        total_formal = sum(score['formal'] for score in all_formality_scores)
        total_informal = sum(score['informal']
                             for score in all_formality_scores)
        avg_formal = total_formal / len(all_formality_scores)
        avg_informal = total_informal / len(all_formality_scores)

        # save the average scores
        avg_line = json.dumps({
            "average_formal": avg_formal,
            "average_informal": avg_informal
        }, ensure_ascii=False)
        f.write(avg_line + '\n')

    print(f"Finish processing and save to {output_filename}")
