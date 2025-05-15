import json
from tqdm import tqdm
import torch

from lm_steer.arguments import parse_args
from lm_steer.models.get_model import get_model


def generate(prompt_data, steer_values, tokenizer, model,
             prompt_num, prompt_length, num_beams, num_beam_groups,
             do_sample, temperature, top_p, device, no_control, prompt_steer_values=None):
    for i, _prompt in enumerate(tqdm(prompt_data)):
        _prompt["generations"] = []
        prompt_text = _prompt["prompt"]["text"]
        token_length = tokenizer(prompt_text,
                                 return_tensors="pt")["input_ids"].shape[1]
        
        # 如果是自适应模式且有prompt_steer_values，则使用对应的steer_values
        current_steer_values = steer_values
        if no_control == True and prompt_steer_values is not None and i < len(prompt_steer_values):
            current_steer_values = prompt_steer_values[i]
        
        # import pdb; pdb.set_trace()
            
        for _i in range(prompt_num):
            output = model.generate(
                prompt_text,
                current_steer_values,
                seed=_i,
                max_length=token_length+prompt_length,
                min_length=token_length+prompt_length,
                num_beams=num_beams,
                num_beam_groups=num_beam_groups,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p
            )
            output = output[len(prompt_text):]
            _prompt["generations"].append({
                "text": output
            })
        if args.verbose:
            print(prompt_text)
            print(_prompt["generations"])


def main(args):
    device = torch.device("cuda:0") if args.cuda else torch.device("cpu")
    model, tokenizer = get_model(
        args.model_name, args.adapted_component, args.adaptor_class,
        args.num_steers,
        args.rank, args.epsilon, args.init_var, args.low_resource_mode)
    model.to_device(device)

    from torch.serialization import add_safe_globals
    from argparse import Namespace
    add_safe_globals([Namespace])
    ckpt = torch.load(args.ckpt_name, weights_only=False)
    model.load_state_dict(ckpt[1])

    # predicting sentences
    with open(args.eval_file, "r") as f:
        prompt_data = list(map(json.loads, f.readlines()))

    model.eval()
    prompt_num = 25
    prompt_length = 20
    if args.eval_size is not None:
        prompt_data = prompt_data[:args.eval_size]
    num_beams = 1
    num_beam_groups = 1
    do_sample = True
    temperature = args.temperature
    # 初始化prompt_steer_values变量，避免UnboundLocalError
    prompt_steer_values = None
    
    # 如果人为指定control值
    if not args.no_control:
        steer_values = list(map(float, args.steer_values)) \
            if args.steer_values is not None else None
    # 自适应
    else:
        # 使用情感分类器对prompt进行分类，根据情感设置不同的steer_values
        from transformers import pipeline
        print("初始化情感分类器...")
        classifier = pipeline('sentiment-analysis', device=device)
        
        # 存储每个prompt的steer_values
        prompt_steer_values = []
        
        print("对prompt进行情感分类...")
        for _prompt in tqdm(prompt_data, desc='情感分类'):
            prompt_text = _prompt["prompt"]["text"]
            # import pdb; pdb.set_trace()
            try:
                # 对prompt进行情感分类
                prediction = classifier(prompt_text, max_length=512)[0]
                # 根据情感设置steer_values
                if prediction["label"] == "POSITIVE":
                    _prompt["sentiment"] = "POSITIVE"
                    prompt_steer_values.append([2, 1])
                else:
                    _prompt["sentiment"] = "NEGATIVE"
                    prompt_steer_values.append([5, 1])
            except Exception as e:
                print(f"情感分类出错: {e}")
                # 默认设置
                _prompt["sentiment"] = "UNKNOWN"
                prompt_steer_values.append([3, 1])
        
        # 使用第一个prompt的steer_values作为默认值
        steer_values = prompt_steer_values[0] if prompt_steer_values else [3, 1]

    generate(prompt_data, steer_values, tokenizer, model, prompt_num,
             prompt_length, num_beams, num_beam_groups, do_sample, temperature,
             args.top_p, device, args.no_control, prompt_steer_values)

    with open(args.output_file, "w") as f:
        for _prompt in prompt_data:
            f.write(json.dumps(_prompt) + "\n")


if __name__ == "__main__":
    args = parse_args()
    main(args)
