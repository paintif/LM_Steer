TRIAL="transfer"
NUM_SAMPLES=500
ADAPTOR_CLASS="multiply"
model_ckpt_list=(
    "gpt2-medium logs/$TRIAL/gpt2-large-to-medium-transfer.pt"
)

# 训练
PYTHONPATH=. python experiments/steer_transfer.py \
    --ckpt_name logs/reproduction/gpt2-large.pt \
    --n_steps 5000 --lr 0.01\
    --model_name gpt2-medium \
    --transfer_from gpt2-large \
    --output_file logs/$TRIAL/gpt2-large-to-medium-transfer.pt

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


# 定义包含一系列 generations_file 名称的数组
generations_file_list=()
for pair in "${model_ckpt_list[@]}"; do
    IFS=' ' read -r MODEL ckpt_name <<< "$pair"
    prefix=$(basename "$ckpt_name" ".pt")
    prefix=${prefix#logs/$TRIAL/}
    generations_file="logs/$TRIAL/${prefix}-predictions-$NUM_SAMPLES.jsonl"
    generations_file_list+=("$generations_file")
done

# 评估生成的文件
echo "evaluating predictions..."
for generations_file in "${generations_file_list[@]}"; do
    base_name=$(basename "$generations_file" ".jsonl")
    output_file="${base_name}-eval.txt"

    PYTHONPATH=. python experiments/evaluation/evaluate.py \
        --generations_file "$generations_file" \
        --metrics ppl-big,dist-n \
        --output_file "$output_file"
done  