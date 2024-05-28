# Evaluation

## opencompass
首先，进入 `vlmevalkit` 目录下，安装必要的依赖：
```bash
cd vlmevalkit
pip install -r requirements.txt
```
<br />

然后，运行 `script/run_inference.sh`，该脚本依次接收三个输入参数：`MODELNAME`, `DATALIST`, `MODE`。`MODELNAME` 为模型名称，`DATALIST` 为目标数据集，`MODE` 为评测模式。
```bash
chmod +x ./script/run_inference.sh
./script/run_inference.sh $MODELNAME $DATALIST $MODE
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
./script/run_inference.sh MiniCPM-Llama3-V-2_5 "MME MMBench_TEST_EN MMBench_TEST_CN MMMU_DEV_VAL MathVista_MINI LLaVABench RealWorldQA" all

# 以下是单独运行 1 个数据集的指令
# MME
./script/run_inference.sh MiniCPM-Llama3-V-2_5 MME all
# MMBench_TEST_EN
./script/run_inference.sh MiniCPM-Llama3-V-2_5 MMBench_TEST_EN all
# MMBench_TEST_CN
./script/run_inference.sh MiniCPM-Llama3-V-2_5 MMBench_TEST_CN all
# MMMU_DEV_VAL
./script/run_inference.sh MiniCPM-Llama3-V-2_5 MMMU_DEV_VAL all
# MathVista_MINI
./script/run_inference.sh MiniCPM-Llama3-V-2_5 MathVista_MINI all
# LLaVABench
./script/run_inference.sh MiniCPM-Llama3-V-2_5 LLaVABench all
# RealWorldQA
./script/run_inference.sh MiniCPM-Llama3-V-2_5 RealWorldQA all
```
<br />

## vqadataset
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

可以传入的参数位于 `eval_utils/getargs.py` 中，各主要参数的含义如下：
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