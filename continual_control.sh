# ./experiments/pipeline.sh {第一个参数（必须）} {第二个参数（可选）}
# 第一个参数说明：
# 0：执行第0部分（准备数据）
# 1：执行第1部分（训练）
# 2：执行第2部分（生成）
# 3：执行第3部分（评估）
# -1：执行所有部分
# 第二个参数说明：
# （可选）针对第二部分代码，指定生成的样本数量，如果不指定则使用原始的10k样本文件
# 检查是否提供了参数

# 若执行时出现语法报错，很有可能是因为换行符错误导致的。换行符有可能为 \r（windows换行符），而shell脚本需要 \n。可以下载dos2unix工具，并执行sed -i 's/\r$//' continual_control.sh  命令进行修复
if [ $# -eq 0 ]; then
    echo "请提供执行部分的第一个参数（0、1、2、3、4或-1）"
    exit 1
fi

PART=$1
TRIAN_MODEL=gpt2-large
# ======detoxicity parameters=============
TRIAL=detoxification-$TRIAN_MODEL
# ======sentiment parameters=============
source=positive
control=-5

# ================detoxification part==========================
# ========================================================
# 执行第0部分，创建logs文件夹
if [ $PART -eq 0 ] || [ $PART -eq -1 ]; then
    echo "执行第0部分..."
    mkdir -p logs/$TRIAL
fi

# ========================================================
# 执行第1部分，训练模型
if [ $PART -eq 1 ] || [ $PART -eq -1 ]; then
    echo "执行第1部分，训练模型..."
    PYTHONPATH=. python experiments/training/train.py \
        --dataset_name toxicity \
        --data_dir data/toxicity/jigsaw-unintended-bias-in-toxicity-classification \
        --ckpt_name logs/$TRIAL/checkpoint.pt \
        --model $TRIAN_MODEL --cuda \
        --adaptor_class multiply --num_steers 2 --dummy_steer 1 --rank 1000 \
        --batch_size 32 --max_length 256 \
        --n_steps 1000 --lr 1e-2
fi

# ========================================================
# 执行第2部分，生成
if [ $PART -eq 2 ] || [ $PART -eq -1 ]; then
    echo "执行第2部分，生成语句..."
    
    # 获取第二个参数（样本数量）
    NUM_SAMPLES=$2
    
    # 如果指定了样本数量
    if [ ! -z "$NUM_SAMPLES" ]; then
        echo "生成 $NUM_SAMPLES 个样本..."
        # 执行采样脚本
        PYTHONPATH=. python data/prompts/sample_prompts.py \
            --num_samples $NUM_SAMPLES \
            --input_file data/prompts/nontoxic_prompts-10k.jsonl \
            --output_file data/prompts/nontoxic_prompts/nontoxic_prompts-$NUM_SAMPLES.jsonl
        
        # 使用生成的样本文件进行生成
        PYTHONPATH=. python experiments/training/generate.py \
            --eval_file data/prompts/nontoxic_prompts/nontoxic_prompts-$NUM_SAMPLES.jsonl \
            --output_file logs/$TRIAL/predictions-$NUM_SAMPLES.jsonl \
            --ckpt_name logs/$TRIAL/checkpoint.pt \
            --model $TRIAN_MODEL --cuda \
            --adaptor_class multiply --num_steers 2 --rank 1000 \
            --max_length 256 --verbose --steer_values 5 1
    else
        # 若未指定采样，使用原始的10k样本文件
        PYTHONPATH=. python experiments/training/generate.py \
            --eval_file data/prompts/nontoxic_prompts-10k.jsonl \
            --output_file logs/$TRIAL/predictions.jsonl \
            --ckpt_name logs/$TRIAL/checkpoint.pt \
            --model $TRIAN_MODEL --cuda \
            --adaptor_class multiply --num_steers 2 --rank 1000 \
            --max_length 256 --verbose --steer_values 5 1
    fi
fi

# ========================================================
# 执行第3部分，评估
if [ $PART -eq 3 ] || [ $PART -eq -1 ]; then
    echo "执行第3部分，评估效果..."
    
    # 获取第二个参数（样本数量）
    NUM_SAMPLES=$2
    
    # 根据是否指定样本数量选择对应的预测文件和结果文件
    if [ ! -z "$NUM_SAMPLES" ]; then
        PRED_FILE="logs/$TRIAL/predictions-$NUM_SAMPLES.jsonl"
        RESULT_FILE="result_stats-$NUM_SAMPLES.txt"
    else
        PRED_FILE="logs/$TRIAL/predictions.jsonl"
        RESULT_FILE="result_stats.txt"
    fi
    
    python experiments/evaluation/evaluate2.py \
        --generations_file $PRED_FILE \
        --metrics toxicity,ppl-big,dist-n \
        --output_file $RESULT_FILE
    echo "Detoxification results:"
    cat logs/$TRIAL/$RESULT_FILE
fi

# ================sentiment part==========================
# ========================================================
# 执行第4部分，训练模型
if [ $PART -eq 4 ] || [ $PART -eq -1 ]; then
    TRIAL=sentiment-$TRIAN_MODEL
    mkdir -p logs/$TRIAL
    PYTHONPATH=. python experiments/training/train.py \
        --dataset_name sentiment-sst5 \
        --ckpt_name logs/$TRIAL/checkpoint.pt \
        --model $TRIAN_MODEL --cuda \
        --adaptor_class multiply --num_steers 2 --dummy_steer 1 --rank 1000 \
        --batch_size 32 --max_length 256 \
        --n_steps 1000 --lr 1e-2 --regularization 1e-6 --epsilon 1e-3
fi

# ========================================================
# 执行第5部分，生成
if [ $PART -eq 5 ] || [ $PART -eq -1 ]; then
    TRIAL=sentiment-$TRIAN_MODEL
    PYTHONPATH=. python experiments/training/generate.py \
    --eval_file data/prompts/sentiment_prompts-10k/${source}_prompts.jsonl \
    --output_file logs/$TRIAL/predictions-${source}_${control}.jsonl \
    --ckpt_name logs/$TRIAL/checkpoint.pt \
    --model $TRIAN_MODEL --cuda \
    --adaptor_class multiply --num_steers 2 --rank 1000 \
    --max_length 256 --verbose --steer_values ${control} 1 --top_p 0.9
fi

# ========================================================
# 执行第6部分，评估模型
if [ $PART -eq 6 ] || [ $PART -eq -1 ]; then
    TRIAL=sentiment-$TRIAN_MODEL
    python experiments/evaluation/evaluate.py \
        --generations_file logs/$TRIAL/predictions-${source}_${control}.jsonl \
        --metrics sentiment,ppl-big,dist-n \
        --output_file result_stats_${source}_${control}.txt
    echo "Sentiment control results:"
    cat logs/$TRIAL/result_stats_${source}_${control}.txt
fi

# ========================================================
# 执行第7部分，探索连续控制
if [ $PART -eq 7 ] || [ $PART -eq -1 ]; then
    echo "执行第7部分，探索sentiment连续控制效果..."
    TRIAL=sentiment-$TRIAN_MODEL
    
    # 设置采样数量
    SEN_SAMPLE_NUM=1000
    
    # 首先进行固定采样
    echo "从10k样本中采样${SEN_SAMPLE_NUM}个固定样本..."
    PYTHONPATH=. python data/prompts/sample_prompts.py \
        --num_samples ${SEN_SAMPLE_NUM} \
        --input_file data/prompts/sentiment_prompts-10k/${source}_prompts.jsonl \
        --output_file data/prompts/sentiment_prompts-10k/sampled_${SEN_SAMPLE_NUM}_${source}_prompts.jsonl \
        --seed 42
    
    # 使用不同的steer values对采样后的prompts进行生成
    for steer_value in -4 ; do
        echo "使用 sentiment steer_value: $steer_value"
        PYTHONPATH=. python experiments/training/generate.py \
            --eval_file data/prompts/sentiment_prompts-10k/sampled_${SEN_SAMPLE_NUM}_${source}_prompts.jsonl \
            --output_file logs/$TRIAL/predictions_continuous_${SEN_SAMPLE_NUM}_${steer_value}.jsonl \
            --ckpt_name logs/$TRIAL/checkpoint.pt \
            --model $TRIAN_MODEL --cuda \
            --adaptor_class multiply --num_steers 2 --rank 1000 \
            --max_length 256 --verbose --steer_values ${steer_value} 1 --top_p 0.9
            
        # 评估每个steer value的效果
        python experiments/evaluation/evaluate.py \
            --generations_file logs/$TRIAL/predictions_continuous_${SEN_SAMPLE_NUM}_${steer_value}.jsonl \
            --metrics sentiment,ppl-big,dist-n \
            --output_file result_stats_continuous_${SEN_SAMPLE_NUM}_${steer_value}.txt
        echo "Continuous sentiment control results (steer_value = ${steer_value}):"
        cat logs/$TRIAL/result_stats_continuous_${SEN_SAMPLE_NUM}_${steer_value}.txt
    done
fi

# ========================================================
# 执行第8部分，探索组合效果
if [ $PART -eq 8 ] || [ $PART -eq -1 ]; then
    echo "执行第8部分，探索detoxification和sentiment的组合效果..."
    TRIAL=combined-$TRIAN_MODEL
    
    # 首先组合两个控制器
    echo "组合detoxification和sentiment控制器..."
    PYTHONPATH=. python lm_steer/combine_steers.py --model $TRIAN_MODEL
    
    # 设置采样数量
    COMB_SAMPLE_NUM=500
    
    # 进行固定采样
    echo "从10k样本中采样${COMB_SAMPLE_NUM}个固定样本..."
    PYTHONPATH=. python data/prompts/sample_prompts.py \
        --num_samples ${COMB_SAMPLE_NUM} \
        --input_file data/prompts/nontoxic_prompts-10k.jsonl \
        --output_file data/prompts/nontoxic_prompts/sampled_${COMB_SAMPLE_NUM}_prompts.jsonl \
        --seed 42

    # de = 3 ,sen = 5还没有跑出来

    # 探索不同组合的效果
    for detox_value in 0 1 2 3 4 5; do
        for sent_value in -5 -3 -1 0 1 3 5; do
            echo "使用组合: detox=${detox_value}, sentiment=${sent_value}"
            PYTHONPATH=. python experiments/training/generate.py \
                --eval_file data/prompts/nontoxic_prompts/sampled_${COMB_SAMPLE_NUM}_prompts.jsonl \
                --output_file logs/$TRIAL/predictions_detox${detox_value}_sent${sent_value}.jsonl \
                --ckpt_name logs/$TRIAL/checkpoint.pt \
                --model $TRIAN_MODEL --cuda \
                --adaptor_class multiply --num_steers 4 --rank 1000 \
                --max_length 256 --verbose --steer_values ${detox_value} 1 ${sent_value} 1
                
            # 评估每个组合的效果
            python experiments/evaluation/evaluate.py \
                --generations_file logs/$TRIAL/predictions_detox${detox_value}_sent${sent_value}.jsonl \
                --metrics sentiment,ppl-big,dist-n \
                --output_file result_stats_detox${detox_value}_sent${sent_value}.txt
            echo "Combined control results (detox=${detox_value}, sentiment=${sent_value}):"
            cat logs/$TRIAL/result_stats_detox${detox_value}_sent${sent_value}.txt
        done
    done
fi

# ========================================================
# 执行第9部分，对combine目录下的结果进行毒性测试
if [ $PART -eq 9 ] || [ $PART -eq -1 ]; then
    echo "执行第9部分，对combine目录下的结果进行毒性测试..."
    TRIAL=combined-$TRIAN_MODEL
    mkdir -p logs/$TRIAL
    
    # 遍历combine目录下的所有jsonl文件
    for file in logs/$TRIAL/predictions_*.jsonl; do
        if [ -f "$file" ]; then
            echo "正在评估文件: $file"
            # 获取不带路径的文件名
            filename=$(basename "$file" .jsonl)
            
            # 评估毒性
            python experiments/evaluation/evaluate.py \
                --generations_file "$file" \
                --metrics toxicity \
                --output_file "${filename}.txt"
            echo "评估结果已保存至: logs/$TRIAL/${filename}.txt"
            echo "评估结果:"
            cat "logs/$TRIAL/${filename}.txt"
            echo "----------------------------------------"
        fi
    done
fi

# toxicity,