import os
import re
import csv

# DIR = "../logs/combined-gpt2-large2"
DIR = "../logs/combined-gpt2-large"
MODEL_NAME = "gpt2-large"
OUTPUT_CSV = "../logs/csv_data/combine.csv"

FIELDS = [
    "model",
    "detox",
    "sent",
    "gpt2-xl perplexity",
    "gpt2-xl total perplexity",
    "mean sentiment accuracy",
    "sentiment accuracy std",
    "dist-1",
    "dist-2",
    "dist-3",
    "avg_max",
    "toxicity prob"
]

def parse_filename(filename):
    # 只要包含 detoxX_sentY.txt 即可，不要求前缀
    m = re.search(r"detox(\d+)_sent(-?\d+)\.txt$", filename)
    if m:
        detox = int(m.group(1))
        sent = int(m.group(2))
        return detox, sent
    return None, None

def parse_file(filepath):
    result = {}
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            # 支持一行多种数据同时提取
            if line.startswith("gpt2-xl perplexity"):
                m = re.search(r"= ([\d\.eE+-]+), ([\d\.eE+-]+)", line)
                if m:
                    result["gpt2-xl perplexity"] = float(m.group(1))
                    result["gpt2-xl total perplexity"] = float(m.group(2))
            if line.startswith("mean sentiment accuracy"):
                m = re.search(r"= ([\d\.eE+-]+), ([\d\.eE+-]+)", line)
                if m:
                    result["mean sentiment accuracy"] = float(m.group(1))
                    result["sentiment accuracy std"] = float(m.group(2))
            if line.startswith("dist-1"):
                m = re.search(r"= ([\d\.eE+-]+)", line)
                if m:
                    result["dist-1"] = float(m.group(1))
            if line.startswith("dist-2"):
                m = re.search(r"= ([\d\.eE+-]+)", line)
                if m:
                    result["dist-2"] = float(m.group(1))
            if line.startswith("dist-3"):
                m = re.search(r"= ([\d\.eE+-]+)", line)
                if m:
                    result["dist-3"] = float(m.group(1))
            # 支持一行多指标
            m = re.search(r"avg_max\s*=\s*([\d\.eE+-]+)", line)
            if m:
                result["avg_max"] = float(m.group(1))
            m2 = re.search(r"toxicity prob\s*=\s*([\d\.eE+-]+)", line)
            if m2:
                result["toxicity prob"] = float(m2.group(1))
    return result

def main():
    # 读取已有csv内容（如果存在）
    rows = []
    key_map = {}
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 保证所有字段都在
                for field in FIELDS:
                    if field not in row:
                        row[field] = ""
                # 统一类型
                row["detox"] = int(row["detox"]) if row["detox"] != "" else ""
                row["sent"] = int(row["sent"]) if row["sent"] != "" else ""
                key = (row["model"], row["detox"], row["sent"])
                rows.append(row)
                key_map[key] = row

    # 遍历txt文件，合并数据
    for fname in os.listdir(DIR):
        # 只要结尾是 detoxX_sentY.txt 就处理
        if re.search(r"detox\d+_sent-?\d+\.txt$", fname) and "ppl-big" not in fname and "sentiment" not in fname:
            detox, sent = parse_filename(fname)
            if detox is None or sent is None:
                continue
            file_path = os.path.join(DIR, fname)
            metrics = parse_file(file_path)
            key = (MODEL_NAME, detox, sent)
            if key in key_map:
                row = key_map[key]
            else:
                row = {field: "" for field in FIELDS}
                row["model"] = MODEL_NAME
                row["detox"] = detox
                row["sent"] = sent
                rows.append(row)
                key_map[key] = row
            # 更新对应列
            for k, v in metrics.items():
                if k in FIELDS:
                    row[k] = v

    # 按detox和sent排序
    rows.sort(key=lambda x: (x["detox"], x["sent"]))
    # 写入csv
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"已保存到 {OUTPUT_CSV}")

if __name__ == "__main__":
    main()