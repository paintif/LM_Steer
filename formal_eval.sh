python eval-process.py -i D:\Desktop\origin\gpt2-origin-predictions-500.jsonl -o D:\Desktop\origin\gpt2-process-origin-predictions-500.jsonl

# evaluate formality
PYTHONPATH=. python experiments/evaluation/eval_formal.py \
    --generations_file logs/formality/gpt2-medium-process-formal-1-neg-predictions-500.jsonl \
    --metrics formality \
    --output_file eval-gpt2-medium-formal-1-neg-predictions-500.jsonl