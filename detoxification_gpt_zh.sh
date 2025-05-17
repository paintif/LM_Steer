TRIAL=detoxification-gpt2-large-zh
mkdir -p logs/$TRIAL
PYTHONPATH=. python experiments/training/train.py \
    --dataset_name toxicity_zh \
    --data_dir data/toxicity/jigsaw-unintended-bias-in-toxicity-classification \
    --ckpt_name logs/$TRIAL/checkpoint.pt \
    --model uer/gpt2-large-chinese-cluecorpussmall --cuda \
    --adaptor_class multiply --num_steers 2 --dummy_steer 1 --rank 1000 \
    --batch_size 32 --max_length 256 \
    --n_steps 1000 --lr 1e-2


PYTHONPATH=. python experiments/training/generate.py \
    --eval_file data/prompts/nontoxic_prompts-10k_zh.jsonl \
    --output_file logs/$TRIAL/predictions.jsonl \
    --ckpt_name logs/$TRIAL/checkpoint.pt \
    --model uer/gpt2-large-chinese-cluecorpussmall --cuda \
    --adaptor_class multiply --num_steers 2 --rank 1000 \
    --max_length 256 --verbose --steer_values 5 1

python experiments/evaluation/evaluate.py \
    --generations_file logs/$TRIAL/predictions.jsonl \
    --metrics toxicity,ppl-zh,dist-n \
    --output_file result_stats.txt
echo "Detoxification zh results:"
cat logs/$TRIAL/result_stats.txt