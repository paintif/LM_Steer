NUM_SAMPLES=500
#生成样本提示文件
echo "generating samples..." 
PYTHONPATH=. python data/prompts/sample_prompts.py \
    --num_samples $NUM_SAMPLES \
    --input_file data/prompts/nontoxic_prompts-10k.jsonl \
    --output_file data/prompts/nontoxic_prompts/sampled_${NUM_SAMPLES}_prompts.jsonl