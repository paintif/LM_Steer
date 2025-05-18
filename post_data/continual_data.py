import os
import re
import csv

DIR = "../logs/sentiment-gpt2-large"
MODEL_NAME = "gpt2-large"
OUTPUT_CSV = "../logs/csv_data/continual.csv"

FIELDS = [
    "model",
    "sent",
    "gpt2-xl perplexity",
    "gpt2-xl total perplexity",
    "mean sentiment accuracy",
    "sentiment accuracy std",
    "dist-1",
    "dist-2",
    "dist-3"
]

def parse_filename(filename):
    # 匹配 result_stats_continuous_1000_5.txt，提取最后的sent值
    m = re.search(r"result_stats_continuous_\d+_(-?\d+)\.txt$", filename)
    if m:
        sent = int(m.group(1))
        return sent
    return None

def parse_file(filepath):
    result = {}
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
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
    return result

def main():
    rows = []
    key_map = {}
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for field in FIELDS:
                    if field not in row:
                        row[field] = ""
                row["sent"] = int(row["sent"]) if row["sent"] != "" else ""
                key = (row["model"], row["sent"])
                rows.append(row)
                key_map[key] = row

    for fname in os.listdir(DIR):
        # 只处理 result_stats_continuous_1000_5.txt 这类文件
        if re.search(r"result_stats_continuous_\d+_-?\d+\.txt$", fname):
            sent = parse_filename(fname)
            if sent is None:
                continue
            file_path = os.path.join(DIR, fname)
            metrics = parse_file(file_path)
            key = (MODEL_NAME, sent)
            if key in key_map:
                row = key_map[key]
            else:
                row = {field: "" for field in FIELDS}
                row["model"] = MODEL_NAME
                row["sent"] = sent
                rows.append(row)
                key_map[key] = row
            for k, v in metrics.items():
                if k in FIELDS:
                    row[k] = v

    rows.sort(key=lambda x: x["sent"])
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"已保存到 {OUTPUT_CSV}")

if __name__ == "__main__":
    main()