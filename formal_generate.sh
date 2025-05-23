TRIAL="formality-2"
NUM_SAMPLES=500
ADAPTOR_CLASS="multiply"
model_ckpt_list=(
    "gpt2-large logs/$TRIAL/gpt2-large-formal-2.pt"
)

# 生成预测文件
echo "generating predictions..."
for pair in "${model_ckpt_list[@]}"; do
    # 拆分二元组
    IFS=' ' read -r MODEL ckpt_name <<< "$pair"
    eval_file="data/prompts/nontoxic_prompts/sampled_${NUM_SAMPLES}_prompts.jsonl"
    prefix=$(basename "$ckpt_name" ".pt")
    prefix=${prefix#logs/$TRIAL/}
    output_file="logs/$TRIAL/${prefix}-predictions-$NUM_SAMPLES.jsonl"

    PYTHONPATH=. python experiments/training/generate.py \
        --eval_file "$eval_file" \
        --output_file "$output_file" \
        --ckpt_name "$ckpt_name" \
        --model "$MODEL" --cuda \
        --adaptor_class $ADAPTOR_CLASS --num_steers 2 --rank 1000 \
        --max_length 256 --verbose --steer_values 0.5 1
done