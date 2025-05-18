import torch
import argparse

def combine_steers(model_name):
    # 加载两个checkpoint
    detox_ckpt = torch.load(f'logs/detoxification-{model_name}/checkpoint.pt', weights_only=False)
    sent_ckpt = torch.load(f'logs/sentiment-{model_name}/checkpoint.pt', weights_only=False)

    # 获取两个控制器的projector矩阵
    detox_proj1 = detox_ckpt[1]['projector1']
    detox_proj2 = detox_ckpt[1]['projector2']
    sent_proj1 = sent_ckpt[1]['projector1']
    sent_proj2 = sent_ckpt[1]['projector2']

    # 打印调试信息
    print("Detox projector shapes:", detox_proj1.shape, detox_proj2.shape)
    print("Sentiment projector shapes:", sent_proj1.shape, sent_proj2.shape)

    # 拼接矩阵
    combined_proj1 = torch.cat([detox_proj1, sent_proj1], dim=0)
    combined_proj2 = torch.cat([detox_proj2, sent_proj2], dim=0)

    # 创建与原始checkpoint相同的格式
    args = detox_ckpt[0]  # 使用detox的参数作为基础
    args.num_steers = 4   # 更新为组合后的steers数量
    
    state_dict = {
        'projector1': combined_proj1,
        'projector2': combined_proj2
    }
    
    rank = detox_ckpt[2]

    # 保存组合后的模型，保持与原始格式一致
    combined_ckpt = [args, state_dict, rank]
    torch.save(combined_ckpt, f'logs/combined-{model_name}/checkpoint.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='模型名称，如gpt2-large')
    args = parser.parse_args()
    
    # 创建保存目录
    import os
    os.makedirs(f'logs/combined-{args.model}', exist_ok=True)
    
    combine_steers(args.model)