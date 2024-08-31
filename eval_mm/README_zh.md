# Evaluation

## MiniCPM-V 2.6

### opencompass
首先，进入 `vlmevalkit` 目录下，安装必要的依赖：
```bash
cd vlmevalkit
pip install --upgrade pip
pip install -e .
wget https://download.pytorch.org/whl/cu118/torch-2.2.0%2Bcu118-cp310-cp310-linux_x86_64.whl#sha256=4377e0a7fe8ff8ffc4f7c9c6130c1dcd3874050ae4fc28b7ff1d35234fbca423
wget https://download.pytorch.org/whl/cu118/torchvision-0.17.0%2Bcu118-cp310-cp310-linux_x86_64.whl#sha256=2e63d62e09d9b48b407d3e1b30eb8ae4e3abad6968e8d33093b60d0657542428
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu118torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install torch-2.2.0%2Bcu118-cp310-cp310-linux_x86_64.whl
pip install torchvision-0.17.0%2Bcu118-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.6.3+cu118torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
rm *.whl
```
<br />

然后，运行 `scripts/run_inference.sh`，该脚本依次接收三个输入参数：`MODELNAME`, `DATALIST`, `MODE`。`MODELNAME` 为模型名称，`DATALIST` 为目标数据集，`MODE` 为评测模式。
```bash
chmod +x ./scripts/run_inference.sh
./scripts/run_inference.sh $MODELNAME $DATALIST $MODE
```
<br />

`MODELNAME` 有四种选择，位于 `vlmeval/config.py` 中：
```bash
minicpm_series = {
    'MiniCPM-V': partial(MiniCPM_V, model_path='openbmb/MiniCPM-V'),
    'MiniCPM-V-2': partial(MiniCPM_V, model_path='openbmb/MiniCPM-V-2'),
    'MiniCPM-Llama3-V-2_5': partial(MiniCPM_Llama3_V, model_path='openbmb/MiniCPM-Llama3-V-2_5'),
    'MiniCPM-V-2_6': partial(MiniCPM_V_2_6, model_path='openbmb/MiniCPM-V-2_6'),
}
```
<br />

可选的所有 `DATALIST` 位于 `vlmeval/utils/dataset_config.py` 中。将不同数据集名称以空格隔开，两端加引号：
```bash
$DATALIST="MMMU_DEV_VAL MathVista_MINI MMVet MMBench_DEV_EN_V11 MMBench_DEV_CN_V11 MMStar HallusionBench AI2D_TEST"
```
<br />

直接对各 benchmark 进行评分时，设置 `MODE=all`。如果仅需要推理结果，则设置 `MODE=infer`。
为了复现出首页展示的表格中的各项结果（MME 到 HallusionBench 之间的列），需要按照如下设置运行：
```bash
# without CoT
./scripts/run_inference.sh MiniCPM-V-2_6 "MMMU_DEV_VAL MathVista_MINI MMVet MMBench_DEV_EN_V11 MMBench_DEV_CN_V11 MMStar HallusionBench AI2D_TEST" all
./scripts/run_inference.sh MiniCPM-V-2_6 MME all
# with CoT，运行 CoT 版本的 MME 时，需要改写 vlmeval/vlm/minicpm_v.py 中的 'use_cot' 函数，将 MME 添加到 return True 的分支中
./scripts/run_inference/sh MiniCPM-V-2_6 "MMMU_DEV_VAL MMVet MMStar HallusionBench OCRBench" all
./scripts/run_inference.sh MiniCPM-V-2_6 MME all
```
<br />

### vqadataset
首先，进入 `vqaeval` 目录下，安装必要的依赖，并创建 `downloads` 子目录，用于存储下载的数据集：
```bash
cd vqaeval
pip install -r requirements.txt
mkdir downloads
```
<br />

然后，从下列各地址下载数据集并置于指定目录下：
###### TextVQA
```bash
cd downloads
mkdir TextVQA && cd TextVQA
wget https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip
unzip train_val_images.zip && rm train_val_images.zip
mv train_val_images/train_images . && rm -rf train_val_images
wget https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json
cd ../..
```

###### DocVQA / DocVQATest
```bash
cd downloads
mkdir DocVQA && cd DocVQA && mkdir spdocvqa_images
# 在 https://rrc.cvc.uab.es/?ch=17&com=downloads 下载 Task 1 - Single Page Document Visual Question Answering 下的 Images 和 Annotations
# 将下载得到的 spdocvqa_images.tar.gz 以及 spdocvqa_qas.zip 置于 DocVQA 目录下
tar -zxvf spdocvqa_images.tar.gz -C spdocvqa_images && rm spdocvqa_images.tar.gz
unzip spdocvqa_qas.zip && rm spdocvqa_qas.zip
cp spdocvqa_qas/val_v1.0_withQT.json . && cp spdocvqa_qas/test_v1.0.json .  && rm -rf spdocvqa_qas
cd ../..
```
<br />

`downloads` 目录应当按照下列结构组织：
```bash
downloads
├── TextVQA
│   ├── train_images
│   │   ├── ...
│   ├── TextVQA_0.5.1_val.json
├── DocVQA
│   ├── spdocvqa_images
│   │   ├── ...
│   ├── val_v1.0_withQT.json
│   ├── test_v1.0.json
```
<br />

准备好相应的数据集之后，修改 `shell/run_inference.sh` 的参数，运行推理：

```bash
chmod +x ./shell/run_inference.sh
./shell/run_inference.sh
```
<br />

可以传入的参数位于 `eval_utils/getargs.py` 中，各主要参数的含义如下。
对于 `MiniCPM-V-2_6`，需要将 `model_name`设置为 `minicpmv26`：
```bash
# 指定 TextVQA 评测所有图片和问题的路径
--textVQA_image_dir
--textVQA_ann_path
# 指定 DocVQA 评测所有图片和问题的路径
--docVQA_image_dir
--docVQA_ann_path
# 指定 DocVQATest 评测所有图片和问题的路径
--docVQATest_image_dir
--docVQATest_ann_path

# 决定是否评测某个任务，eval_all 设置为 True 表示所有任务都评测
--eval_textVQA
--eval_docVQA
--eval_docVQATest
--eval_all

# 模型名称、模型路径（从指定路径加载模型）
--model_name
--model_path
# 从 checkpoint 加载模型
--ckpt
# 模型处理输入数据的方式，interleave 表示图文交错式，old 表示非交错式
--generate_method
# 推理时的批处理规模，建议推理时设置为 1
--batchsize

# 输出内容保存的路径
--answer_path
```
<br />

评测三个任务需要设置的参数如下：
###### TextVQA
```bash
--eval_textVQA
--textVQA_image_dir ./downloads/TextVQA/train_images
--textVQA_ann_path ./downloads/TextVQA/TextVQA_0.5.1_val.json
```

###### DocVQA
```bash
--eval_docVQA
--docVQA_image_dir ./downloads/DocVQA/spdocvqa_images
--docVQA_ann_path ./downloads/DocVQA/val_v1.0_withQT.json
```

###### DocVQATest
```bash
--eval_docVQATest
--docVQATest_image_dir ./downloads/DocVQA/spdocvqa_images
--docVQATest_ann_path ./downloads/DocVQA/test_v1.0.json
```
<br />

对于 DocVQATest 任务，为了将推理结果上传到[官方网站](https://rrc.cvc.uab.es/?ch=17)进行评测，还需要运行 `shell/run_transform.sh` 进行格式转换。其中，`input_file_path` 对应原始输出的 json 的路径，`output_file_path` 为自定义的转换后的 json 的路径：
```bash
chmod +x ./shell/run_transform.sh
./shell/run_transform.sh
```
<br />

## MiniCPM-Llama3-V-2_5

<details>
<summary>展开</summary>

### opencompass
首先，进入 `vlmevalkit` 目录下，安装必要的依赖：
```bash
cd vlmevalkit
pip install -r requirements.txt
```
<br />

然后，运行 `scripts/run_inference.sh`，该脚本依次接收三个输入参数：`MODELNAME`, `DATALIST`, `MODE`。`MODELNAME` 为模型名称，`DATALIST` 为目标数据集，`MODE` 为评测模式。
```bash
chmod +x ./scripts/run_inference.sh
./scripts/run_inference.sh $MODELNAME $DATALIST $MODE
```
<br />

`MODELNAME` 有三种选择，位于 `vlmeval/config.py` 中：
```bash
ungrouped = {
    'MiniCPM-V':partial(MiniCPM_V, model_path='openbmb/MiniCPM-V'),
    'MiniCPM-V-2':partial(MiniCPM_V, model_path='openbmb/MiniCPM-V-2'),
    'MiniCPM-Llama3-V-2_5':partial(MiniCPM_Llama3_V, model_path='openbmb/MiniCPM-Llama3-V-2_5'),
}
```
<br />

可选的所有 `DATALIST` 位于 `vlmeval/utils/dataset_config.py` 中，评测单个数据集时，直接调用数据集名称，不加引号；评测多个数据集时，将不同数据集名称以空格隔开，两端加引号：
```bash
$DATALIST="POPE ScienceQA_TEST ChartQA_TEST"
```
<br />

直接对各 benchmark 进行评分时，设置 `MODE=all`。如果仅需要推理结果，则设置 `MODE=infer`
为了复现出首页展示的表格中的各项结果（MME 到 RealWorldQA 之间的列），需要按照如下设置运行：
```bash
# 一次性运行 7 个数据集
./scripts/run_inference.sh MiniCPM-Llama3-V-2_5 "MME MMBench_TEST_EN MMBench_TEST_CN MMMU_DEV_VAL MathVista_MINI LLaVABench RealWorldQA" all

# 以下是单独运行 1 个数据集的指令
# MME
./scripts/run_inference.sh MiniCPM-Llama3-V-2_5 MME all
# MMBench_TEST_EN
./scripts/run_inference.sh MiniCPM-Llama3-V-2_5 MMBench_TEST_EN all
# MMBench_TEST_CN
./scripts/run_inference.sh MiniCPM-Llama3-V-2_5 MMBench_TEST_CN all
# MMMU_DEV_VAL
./scripts/run_inference.sh MiniCPM-Llama3-V-2_5 MMMU_DEV_VAL all
# MathVista_MINI
./scripts/run_inference.sh MiniCPM-Llama3-V-2_5 MathVista_MINI all
# LLaVABench
./scripts/run_inference.sh MiniCPM-Llama3-V-2_5 LLaVABench all
# RealWorldQA
./scripts/run_inference.sh MiniCPM-Llama3-V-2_5 RealWorldQA all
```
<br />

### vqadataset
首先，进入 `vqaeval` 目录下，安装必要的依赖，并创建 `downloads` 子目录，用于存储下载的数据集：
```bash
cd vqaeval
pip install -r requirements.txt
mkdir downloads
```
<br />

然后，从下列各地址下载数据集并置于指定目录下：
###### TextVQA
```bash
cd downloads
mkdir TextVQA && cd TextVQA
wget https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip
unzip train_val_images.zip && rm train_val_images.zip
mv train_val_images/train_images . && rm -rf train_val_images
wget https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json
cd ../..
```

###### DocVQA / DocVQATest
```bash
cd downloads
mkdir DocVQA && cd DocVQA && mkdir spdocvqa_images
# 在 https://rrc.cvc.uab.es/?ch=17&com=downloads 下载 Task 1 - Single Page Document Visual Question Answering 下的 Images 和 Annotations
# 将下载得到的 spdocvqa_images.tar.gz 以及 spdocvqa_qas.zip 置于 DocVQA 目录下
tar -zxvf spdocvqa_images.tar.gz -C spdocvqa_images && rm spdocvqa_images.tar.gz
unzip spdocvqa_qas.zip && rm spdocvqa_qas.zip
cp spdocvqa_qas/val_v1.0_withQT.json . && cp spdocvqa_qas/test_v1.0.json .  && rm -rf spdocvqa_qas
cd ../..
```
<br />

`downloads` 目录应当按照下列结构组织：
```bash
downloads
├── TextVQA
│   ├── train_images
│   │   ├── ...
│   ├── TextVQA_0.5.1_val.json
├── DocVQA
│   ├── spdocvqa_images
│   │   ├── ...
│   ├── val_v1.0_withQT.json
│   ├── test_v1.0.json
```
<br />

准备好相应的数据集之后，修改 `shell/run_inference.sh` 的参数，运行推理：

```bash
chmod +x ./shell/run_inference.sh
./shell/run_inference.sh
```
<br />

可以传入的参数位于 `eval_utils/getargs.py` 中，各主要参数的含义如下。
对于 `MiniCPM-Llama3-V-2_5`，需要将 `model_name` 设置为 `minicpmv`：
```bash
# 指定 TextVQA 评测所有图片和问题的路径
--textVQA_image_dir
--textVQA_ann_path
# 指定 DocVQA 评测所有图片和问题的路径
--docVQA_image_dir
--docVQA_ann_path
# 指定 DocVQATest 评测所有图片和问题的路径
--docVQATest_image_dir
--docVQATest_ann_path

# 决定是否评测某个任务，eval_all 设置为 True 表示所有任务都评测
--eval_textVQA
--eval_docVQA
--eval_docVQATest
--eval_all

# 模型名称、模型路径（从指定路径加载模型）
--model_name
--model_path
# 从 checkpoint 加载模型
--ckpt
# 模型处理输入数据的方式，interleave 表示图文交错式，old 表示非交错式
--generate_method
# 推理时的批处理规模，建议推理时设置为 1
--batchsize

# 输出内容保存的路径
--answer_path
```
<br />

评测三个任务需要设置的参数如下：
###### TextVQA
```bash
--eval_textVQA
--textVQA_image_dir ./downloads/TextVQA/train_images
--textVQA_ann_path ./downloads/TextVQA/TextVQA_0.5.1_val.json
```

###### DocVQA
```bash
--eval_docVQA
--docVQA_image_dir ./downloads/DocVQA/spdocvqa_images
--docVQA_ann_path ./downloads/DocVQA/val_v1.0_withQT.json
```

###### DocVQATest
```bash
--eval_docVQATest
--docVQATest_image_dir ./downloads/DocVQA/spdocvqa_images
--docVQATest_ann_path ./downloads/DocVQA/test_v1.0.json
```
<br />

对于 DocVQATest 任务，为了将推理结果上传到[官方网站](https://rrc.cvc.uab.es/?ch=17)进行评测，还需要运行 `shell/run_transform.sh` 进行格式转换。其中，`input_file_path` 对应原始输出的 json 的路径，`output_file_path` 为自定义的转换后的 json 的路径：
```bash
chmod +x ./shell/run_transform.sh
./shell/run_transform.sh
```

</details>