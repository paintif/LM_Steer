if [ $# -eq 0 ]; then
    echo "请提供执行部分的第一个参数（0、1、2、3、4或-1）"
    exit 1
fi

PART=$1
TRIAN_MODEL=gpt2-large
# ======detoxicity parameters=============
TRIAL=input-sentiment-$TRIAN_MODEL
# ======sentiment parameters=============
source=positive
control=-5

# 执行第4部分，训练模型
if [ $PART -eq 4 ] || [ $PART -eq -1 ]; then
    TRIAL=input-sentiment-$TRIAN_MODEL
    mkdir -p logs/$TRIAL
    PYTHONPATH=. python experiments/training/train.py \
        --dataset_name sentiment-sst5 \
        --ckpt_name logs/$TRIAL/checkpoint.pt \
        --model $TRIAN_MODEL --cuda \
        --adaptor_class multiply --num_steers 2 --dummy_steer 1 --rank 1000 \
        --batch_size 32 --max_length 256 \
        --n_steps 1000 --lr 1e-2 --regularization 1e-6 --epsilon 1e-3 \
        --adapted_component input_embedding
fi

# # ========================================================
# 执行第5部分，生成
if [ $PART -eq 5 ] || [ $PART -eq -1 ]; then
    TRIAL=input-sentiment-$TRIAN_MODEL
    PYTHONPATH=. python experiments/training/generate.py \
    --eval_file data/prompts/sentiment_prompts-10k/${source}_prompts.jsonl \
    --output_file logs/$TRIAL/predictions-${source}_${control}.jsonl \
    --ckpt_name logs/$TRIAL/checkpoint.pt \
    --model $TRIAN_MODEL --cuda \
    --adaptor_class multiply --num_steers 2 --rank 1000 \
    --max_length 256 --steer_values ${control} 1 --top_p 0.9 
fi
# --verbose
# # ========================================================
# 执行第6部分，评估模型
if [ $PART -eq 6 ] || [ $PART -eq -1 ]; then
    TRIAL=input-sentiment-$TRIAN_MODEL
    
    # 设置采样数量
    SEN_SAMPLE_NUM=50
    
    # 首先进行固定采样
    echo "从10k样本中采样${SEN_SAMPLE_NUM}个固定样本..."
    PYTHONPATH=. python data/prompts/sample_prompts.py \
        --num_samples ${SEN_SAMPLE_NUM} \
        --input_file data/prompts/sentiment_prompts-10k/${source}_prompts.jsonl \
        --output_file data/prompts/sentiment_prompts-10k/sampled_${SEN_SAMPLE_NUM}_${source}_prompts.jsonl \
        --seed 42
fi

# ========================================================
# 执行第7部分，探索连续控制
if [ $PART -eq 7 ] || [ $PART -eq -1 ]; then
    echo "执行第7部分，探索sentiment连续控制效果..."
    TRIAL=input-sentiment-$TRIAN_MODEL
    
    # 设置采样数量
    SEN_SAMPLE_NUM=500
    
    首先进行固定采样
    echo "从10k样本中采样${SEN_SAMPLE_NUM}个固定样本..."
    PYTHONPATH=. python data/prompts/sample_prompts.py \
        --num_samples ${SEN_SAMPLE_NUM} \
        --input_file data/prompts/sentiment_prompts-10k/${source}_prompts.jsonl \
        --output_file data/prompts/sentiment_prompts-10k/sampled_${SEN_SAMPLE_NUM}_${source}_prompts.jsonl \
        --seed 42
    
    # 使用不同的steer values对采样后的prompts进行生成
    for steer_value in 4 5; do
        echo "使用 sentiment steer_value: $steer_value"
        PYTHONPATH=. python experiments/training/generate.py \
            --eval_file data/prompts/sentiment_prompts-10k/sampled_${SEN_SAMPLE_NUM}_${source}_prompts.jsonl \
            --output_file logs/$TRIAL/predictions_continuous_${SEN_SAMPLE_NUM}_${steer_value}.jsonl \
            --ckpt_name logs/$TRIAL/checkpoint.pt \
            --model $TRIAN_MODEL --cuda \
            --adaptor_class multiply --num_steers 2 --rank 1000 \
            --max_length 256 --steer_values ${steer_value} 1 --top_p 0.9 \
            # --adapted_component input_embedding
            # --verbose
        # 评估每个steer value的效果
        python experiments/evaluation/evaluate.py \
            --generations_file logs/$TRIAL/predictions_continuous_${SEN_SAMPLE_NUM}_${steer_value}.jsonl \
            --metrics sentiment,ppl-big,dist-n \
            --output_file result_stats_continuous_${SEN_SAMPLE_NUM}_${steer_value}.txt \
            # --adapted_component input_embedding
        echo "Continuous sentiment control results (steer_value = ${steer_value}):"
        cat logs/$TRIAL/result_stats_continuous_${SEN_SAMPLE_NUM}_${steer_value}.txt
    done
fi
