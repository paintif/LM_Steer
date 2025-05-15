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