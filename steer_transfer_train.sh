PYTHONPATH=. python experiments/steer_transfer.py \
    --ckpt_name logs/reproduction/gpt2-large.pt \
    --n_steps 5000 --lr 0.01\
    --model_name gpt2-medium \
    --transfer_from gpt2-large \
    --output_file logs/$TRIAL/gpt2-large-to-medium-transfer.pt