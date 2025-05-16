PYTHONPATH=. python experiments/training/train.py \
    --dataset_name toxicity \
    --data_dir data/formality-2 \
    --ckpt_name logs/formality-2/gpt2-large-formal-2.pt \
    --model gpt2-large --cuda \
    --adaptor_class multiply --num_steers 2 --dummy_steer 1 --rank 1000 \
    --batch_size 32 --max_length 256 \
    --n_steps 1000 --lr 1e-2