# python eval-process.py -i D:\Desktop\origin\gpt2-origin-predictions-500.jsonl -o D:\Desktop\origin\gpt2-process-origin-predictions-500.jsonl
PYTHONPATH=. python experiments/evaluation/eval_process.py \
    -i logs/formality-2/gpt2-large-formal-2-predictions-500.jsonl \
    -o logs/formality-2/gpt2-large-process-formal-2-predictions-500.jsonl

# evaluate formality
PYTHONPATH=. python experiments/evaluation/eval_formal.py \
    --generations_file logs/formality-2/gpt2-large-process-formal-2-predictions-500.jsonl \
    --metrics formality \
    --output_file eval-gpt2-large-formal-2-predictions-500.jsonl.jsonl