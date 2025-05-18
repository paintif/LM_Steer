## LM-Steer 论文复现与拓展实验

**本实验修改自**：https://github.com/Glaciohound/LM-Steer.git

本实验共包含以下部分：
1. 毒性控制实验复现
2. 情感控制复现
3. 模型迁移实验复现
4. 连续控制实验
5. 协同控制实验
6. 风格迁移实验
7. 中文扩展实验
8. 输入层词嵌入实验
9. 自适应情感控制
10. 非线性的控制器

其中1-5为论文复现部分，6-10为创新拓展部份。

实验作者（按名字字母排序）
1. 蔡嘉豪 南京大学
2. 何志烨 南京大学
3. 邱嘉彬 南京大学
4. 郑儒杰 南京大学
### **实验流程总览** 

所有实验均包含 **train（训练）、generate（生成）、evaluate（评估）** 三个核心阶段。其中为了节约 generate 的时间，可以执行采样脚本（data/prompts/sample_prompts.py），指定采样数量以抽取部分数据用于生成，快速验证结果。 
#### 1. **训练阶段（Train）**
- **数据格式**：不同实验的训练数据集格式各异，细节无需深入。 
	- **毒性控制实验**：数据集下载至本地存储。
	- **情感控制实验**：通过 API 远程访问数据集。
	- **其他**：包含小组自主整理的定制化数据集。
#### 2. **生成阶段（Generate）** 
- **核心逻辑**：基于 `prompt` 生成语句，再对生成内容进行评估。
- **数据格式**：生成阶段数据集多以 `prompt.json` 格式存在，用于提供生成引导文本。
- **关键参数**： 
	- `dataset_name`：指定加载的数据集，支持的数据集可通过代码 `LM-Steer\experiments\training\data.py` 中的 `load_dataset` 函数查看。
	- `data_dir`：指定数据集文件路径，但部分远程数据集（如 `sst-5`）无需此参数。 
- **数据采样脚本功能** 
	- **用途**：从 `prompt` 数据集中随机抽取指定数量的数据，用于生成阶段，避免全量数据生成的耗时问题，加速实验迭代。 
	- **执行方式**：直接运行脚本并指定采样数量（具体参数见脚本说明）。
#### 3. **生成阶段（evaluate）** 
#### 1. `eval_file` 
- **作用**：生成阶段的输入文件，存放用于引导生成的 `prompts` 路径。
- **格式**：通常为 `prompt.json` 等结构化文件。
#### 2. `output_file` 
- **作用**：生成阶段的输出文件，用于保存模型生成的语句内容。 **注意该参数不需要额外指定文件夹，默认和`eval_file`处于同一个文件夹下。**
#### 3. `metrics`
- 指定需要评估哪些指标，例如 `dist-n` 说明要评估生成文本的多样性。具体查看 `LM-Steer\experiments\evaluation\evaluate.py`文件 

### **支持的模型与参数规范** 
#### 1. **模型参数（`model`）**
- **功能**：通过 `model` 参数指定实验使用的模型（如 `gpt2-medium`、`gpt2-small` 等）。
- **文件规范**：不同模型的训练 checkpoint 和日志文件需存放在独立文件夹中，命名可参考 `ckpt_name`、`output_file` 等参数的命名规则。
- **模型列表**：支持的模型可通过 `get_module.py` 文件查看。 
#### 2. **Adaptor 类型（`adaptor_class`）** 
- **功能**：指定 `LM-Steer` 模块类型，控制模型的适配逻辑。
- **参数范围**：支持的参数值定义在 `LM-Steer\lm_steer\models\steers.py` 文件中。
- **注意事项**：训练阶段（Train）与生成阶段（Generate）的 `adaptor_class` 需保持一致，确保模型适配逻辑统一。

### 毒性控制实验复现
#### 训练
```bash
./detoxification_sentiment_continual_control.sh 1
```
#### 生成
```bash
./detoxification_sentiment_continual_control.sh 2 [采样数量]
```
#### 评估
```bash
./detoxification_sentiment_continual_control.sh 3
```
- **训练数据集**：`data/toxicity/jigsaw-unintended-bias-in-toxicity-classification`
- **生成和评估的文件路径**：`logs/detoxification-gpt2-large`
- **默认模型**：`gpt2-large`（可自行指定）
- **adaptor_class 默认值**：`multiply`（可自行指定）
- **默认采样脚本启用**，采样数量为500。若要跑全量测试，则使用`data/prompts/nontoxic_prompts-10k.jsonl`引导数据集，注意修改相关参数。
- **默认评估指标**：`toxicity,ppl-big,dist-n`
### 情感控制复现
#### 运行
```bash
LM-Steer/sentiment_control.sh
```
#### 参数指定
- **source**：指定使用什么倾向的prompts来generate（`positive`, `neutral`, `negative`）
- **control**：指定模型生成的情感倾向，分为两档 `-5`（negative）和 `5`（positive）。
- **训练数据集**：`sentiment-sst5`（远程数据集）
- **引导词数据集**：`data/prompts/sentiment_prompts-10k/${source}_prompts.jsonl`
- **默认不开启数据采样脚本**
- **模型**：默认使用`gpt2-large`，可以自行指定
- **评估指标**：`sentiment,ppl-big,dist-n`
- **adaptor_class 默认值**：`multiply`（可自行指定）
- **生成的checkpoint和相关的文件路径**：`logs/sentiment-gpt2-large`
### 模型迁移实验复现
#### 运行
```bash
LM-Steer/steer_transfer.sh
```
#### 参数修改
- **ckpt_name**：训练好的，等待被迁移的模型路径
- **output_file**：目标模型的路径
- **其余参数按命名规范修改**，具体请查阅相关代码
### 连续控制实验
#### 执行
```bash
./detoxification_sentiment_continual_control.sh 7
```
- **默认进行数据采样**，采样数可以在shell脚本中修改
- **默认对情感控制进行连续控制实验**，steer values可以自行指定。其余参数作用和之前的实验类似。
### 协同控制实验
#### 执行
```bash
./detoxification_sentiment_continual_control.sh 8
```
- **实验参数和连续控制实验类似**，不再赘述。
### 风格迁移实验
#### 训练
```bash
formal_train.sh
```
#### 生成
```bash
formal_generate.sh
```
#### 评估
```bash
formal_evaluate.sh
```
- **参数和之前实验类似**
- **特别注意**：`dataset_name` 该参数指定为`toxicity`，是因为该实验的数据集是组内自己整理的，数据处理格式按照`toxicity`数据集的格式处理。将`dataset_name` 该参数指定为`toxicity`可以借用现成的数据处理代码。
### 中文扩展实验
#### 运行
```bash
detoxification_gpt_zh.sh
```
- **这里的模型需要使用支持中文的模型**，默认使用`uer/gpt2-large-chinese-cluecorpussmall`，可以选用Qwen模型。
- **默认使用毒性控制数据集**，可以选用情感控制。
- **其余参数与其他实验类似**。
### 输入层词嵌入实验
#### 运行
```bash
input_ex.sh
```
- **具体运行参数（4，5， 6 ， 7）请阅读注释**
- **默认在情感控制数据集上进行试验**。脚本参数和上面的实验类似，不在赘述。
### 自适应情感控制
#### 运行
```bash
auto_sentiment_control.sh
```
- **不需要修改任何参数**，直接运行即可。
### 非线性的控制器
#### 运行
```bash
nonlinear_sentiment_control.sh
```
- **默认在情感控制数据集上进行试验**
- **参数和情感控制复现实验类似**，不再赘述。
- **注意**：`adaptor_class` 需指定为`nonlinear`
