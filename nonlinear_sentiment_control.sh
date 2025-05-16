
TRIAL=sentiment-gpt2-large-nonlinear
mkdir -p logs/$TRIAL

source=positive
control=-5
PYTHONPATH=. python experiments/training/train.py \
    --dataset_name sentiment-sst5 \
    --ckpt_name logs/$TRIAL/checkpoint.pt \
    --model gpt2-large --cuda \
    --adaptor_class nonlinear --num_steers 2 --dummy_steer 1 --rank 1000 \
    --batch_size 32 --max_length 256 \
    --n_steps 1000 --lr 1e-2 --regularization 1e-6 --epsilon 1e-3
PYTHONPATH=. python experiments/training/generate.py \
    --eval_file data/prompts/sentiment_prompts-10k/${source}_prompts.jsonl \
    --output_file logs/$TRIAL/predictions-${source}_${control}.jsonl \
    --ckpt_name logs/$TRIAL/checkpoint.pt \
    --model gpt2-large --cuda \
    --adaptor_class nonlinear --num_steers 2 --rank 1000 \
    --max_length 256 --verbose --steer_values ${control} 1 --top_p 0.9


python experiments/evaluation/evaluate.py \
    --generations_file logs/$TRIAL/predictions-${source}_${control}.jsonl \
    --metrics sentiment,ppl-big,dist-n \
    --output_file result_stats_${source}_${control}.txt
echo "Sentiment control results:"
cat logs/$TRIAL/result_stats_${source}_${control}.txt
