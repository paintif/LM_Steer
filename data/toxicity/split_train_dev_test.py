import sys 
import random
import numpy as np

train = float(sys.argv[3])
dev = float(sys.argv[4])
test = float(sys.argv[5])
print(train, dev, test)
# 检查比例之和是否为零
total = train + dev + test
if total == 0:
    print("错误: 比例之和不能为零")
    sys.exit(1)
# 归一化比例
train, dev, test = np.array([train, dev, test])/total

try:
    alldata = [line for line in open(sys.argv[1], "r")]
except FileNotFoundError:
    print(f"错误: 找不到文件 '{sys.argv[1]}'")
    sys.exit(1)

random.shuffle(alldata)

# 确保不会出现整数溢出
train_size = int(len(alldata)*train)
dev_size = int(len(alldata)*dev)
test_size = len(alldata) - train_size - dev_size

train_data = alldata[:train_size]
dev_data = alldata[train_size:train_size+dev_size]
test_data = alldata[train_size+dev_size:]

with open(f"{sys.argv[6]}/train_{sys.argv[2]}.jsonl", "w") as fout:
    for line in train_data:
        fout.write(line)

with open(f"{sys.argv[6]}/dev_{sys.argv[2]}.jsonl", "w") as fout:
    for line in dev_data:
        fout.write(line)

with open(f"{sys.argv[6]}/test_{sys.argv[2]}.jsonl", "w") as fout:
    for line in test_data:
        fout.write(line)

