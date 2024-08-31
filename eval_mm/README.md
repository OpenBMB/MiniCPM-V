# Evaluation

## MiniCPM-V 2.6

### opencompass
First, enter the `vlmevalkit` directory and install all dependencies:
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
```
<br />

Then, run `scripts/run_inference.sh`, which receives three input parameters in sequence: `MODELNAME`, `DATALIST`, and `MODE`. `MODELNAME` represents the name of the model, `DATALIST` represents the datasets used for inference, and `MODE` represents evaluation mode:
```bash
chmod +x ./scripts/run_inference.sh
./scripts/run_inference.sh $MODELNAME $DATALIST $MODE
```
<br />

The four available choices for `MODELNAME` are listed in `vlmeval/config.py`:
```bash
minicpm_series = {
    'MiniCPM-V': partial(MiniCPM_V, model_path='openbmb/MiniCPM-V'),
    'MiniCPM-V-2': partial(MiniCPM_V, model_path='openbmb/MiniCPM-V-2'),
    'MiniCPM-Llama3-V-2_5': partial(MiniCPM_Llama3_V, model_path='openbmb/MiniCPM-Llama3-V-2_5'),
    'MiniCPM-V-2_6': partial(MiniCPM_V_2_6, model_path='openbmb/MiniCPM-V-2_6'),
}
```
<br />

All available choices for `DATALIST` are listed in `vlmeval/utils/dataset_config.py`. Separate the names of different datasets with spaces and add quotation marks at both ends:
```bash
$DATALIST="MMMU_DEV_VAL MathVista_MINI MMVet MMBench_DEV_EN_V11 MMBench_DEV_CN_V11 MMStar HallusionBench AI2D_TEST"
```
<br />

While scoring on each benchmark directly, set `MODE=all`. If only inference results are required, set `MODE=infer`. In order to reproduce the results in the table displayed on the homepage (columns between MME and HallusionBench), you need to run the script according to the following settings:
```bash
# without CoT
./scripts/run_inference.sh MiniCPM-V-2_6 "MMMU_DEV_VAL MathVista_MINI MMVet MMBench_DEV_EN_V11 MMBench_DEV_CN_V11 MMStar HallusionBench AI2D_TEST" all
./scripts/run_inference.sh MiniCPM-V-2_6 MME all
# with CoT
# While running the CoT version of MME, you need to modify the 'use_cot' function in vlmeval/vlm/minicpm_v.py and add MME to the branch that returns True.
./scripts/run_inference/sh MiniCPM-V-2_6 "MMMU_DEV_VAL MMVet MMStar HallusionBench OCRBench" all
./scripts/run_inference.sh MiniCPM-V-2_6 MME all
```
<br />

### vqadataset
First, enter the `vqaeval` directory and install all dependencies. Then, create `downloads` subdirectory to store the downloaded dataset for all tasks:
```bash
cd vqaeval
pip install -r requirements.txt
mkdir downloads
```
<br />

Download the datasets from the following links and place it in the specified directories:
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
# Download Images and Annotations from Task 1 - Single Page Document Visual Question Answering at https://rrc.cvc.uab.es/?ch=17&com=downloads
# Move the spdocvqa_images.tar.gz and spdocvqa_qas.zip to DocVQA directory
tar -zxvf spdocvqa_images.tar.gz -C spdocvqa_images && rm spdocvqa_images.tar.gz
unzip spdocvqa_qas.zip && rm spdocvqa_qas.zip
cp spdocvqa_qas/val_v1.0_withQT.json . && cp spdocvqa_qas/test_v1.0.json .  && rm -rf spdocvqa_qas
cd ../..
```
<br />

The `downloads` directory should be organized according to the following structure:
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

Modify the parameters in `shell/run_inference.sh` and run inference:

```bash
chmod +x ./shell/run_inference.sh
./shell/run_inference.sh
```
<br />

All optional parameters are listed in `eval_utils/getargs.py`. The meanings of some major parameters are listed as follows.
For `MiniCPM-V-2_6`, set `model_name` to `minicpmv26`:
```bash
# path to images and their corresponding questions
# TextVQA
--textVQA_image_dir
--textVQA_ann_path
# DocVQA
--docVQA_image_dir
--docVQA_ann_path
# DocVQATest
--docVQATest_image_dir
--docVQATest_ann_path

# whether to eval on certain task
--eval_textVQA
--eval_docVQA
--eval_docVQATest
--eval_all

# model name and model path
--model_name
--model_path
# load model from ckpt
--ckpt
# the way the model processes input data, "interleave" represents interleaved image-text form, while "old" represents non-interleaved.
--generate_method

--batchsize

# path to save the outputs
--answer_path
```
<br />

While evaluating on different tasks, parameters need to be set as follows:
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

For the DocVQATest task, in order to upload the inference results to the [official website](https://rrc.cvc.uab.es/?ch=17) for evaluation, run `shell/run_transform.sh` for format transformation after inference. `input_file_path` represents the path to the original output json, `output_file_path` represents the path to the transformed json:
```bash
chmod +x ./shell/run_transform.sh
./shell/run_transform.sh
```
<br />

## MiniCPM-Llama3-V-2_5

<details>
<summary>Expand</summary>

### opencompass
First, enter the `vlmevalkit` directory and install all dependencies:
```bash
cd vlmevalkit
pip install -r requirements.txt
```
<br />

Then, run `scripts/run_inference.sh`, which receives three input parameters in sequence: `MODELNAME`, `DATALIST`, and `MODE`. `MODELNAME` represents the name of the model, `DATALIST` represents the datasets used for inference, and `MODE` represents evaluation mode:
```bash
chmod +x ./scripts/run_inference.sh
./scripts/run_inference.sh $MODELNAME $DATALIST $MODE
```
<br />

The three available choices for `MODELNAME` are listed in `vlmeval/config.py`:
```bash
ungrouped = {
    'MiniCPM-V':partial(MiniCPM_V, model_path='openbmb/MiniCPM-V'),
    'MiniCPM-V-2':partial(MiniCPM_V, model_path='openbmb/MiniCPM-V-2'),
    'MiniCPM-Llama3-V-2_5':partial(MiniCPM_Llama3_V, model_path='openbmb/MiniCPM-Llama3-V-2_5'),
}
```
<br />

All available choices for `DATALIST` are listed in `vlmeval/utils/dataset_config.py`. While evaluating on a single dataset, call the dataset name directly without quotation marks; while evaluating on multiple datasets, separate the names of different datasets with spaces and add quotation marks at both ends:
```bash
$DATALIST="POPE ScienceQA_TEST ChartQA_TEST"
```
<br />

While scoring on each benchmark directly, set `MODE=all`. If only inference results are required, set `MODE=infer`. In order to reproduce the results in the table displayed on the homepage (columns between MME and RealWorldQA), you need to run the script according to the following settings:
```bash
# run on all 7 datasets
./scripts/run_inference.sh MiniCPM-Llama3-V-2_5 "MME MMBench_TEST_EN MMBench_TEST_CN MMMU_DEV_VAL MathVista_MINI LLaVABench RealWorldQA" all

# The following are instructions for running on a single dataset
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
First, enter the `vqaeval` directory and install all dependencies. Then, create `downloads` subdirectory to store the downloaded dataset for all tasks:
```bash
cd vqaeval
pip install -r requirements.txt
mkdir downloads
```
<br />

Download the datasets from the following links and place it in the specified directories:
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
# Download Images and Annotations from Task 1 - Single Page Document Visual Question Answering at https://rrc.cvc.uab.es/?ch=17&com=downloads
# Move the spdocvqa_images.tar.gz and spdocvqa_qas.zip to DocVQA directory
tar -zxvf spdocvqa_images.tar.gz -C spdocvqa_images && rm spdocvqa_images.tar.gz
unzip spdocvqa_qas.zip && rm spdocvqa_qas.zip
cp spdocvqa_qas/val_v1.0_withQT.json . && cp spdocvqa_qas/test_v1.0.json .  && rm -rf spdocvqa_qas
cd ../..
```
<br />

The `downloads` directory should be organized according to the following structure:
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

Modify the parameters in `shell/run_inference.sh` and run inference:

```bash
chmod +x ./shell/run_inference.sh
./shell/run_inference.sh
```
<br />

All optional parameters are listed in `eval_utils/getargs.py`. The meanings of some major parameters are listed as follows.
For `MiniCPM-Llama3-V-2_5`, set `model_name` to `minicpmv`:
```bash
# path to images and their corresponding questions
# TextVQA
--textVQA_image_dir
--textVQA_ann_path
# DocVQA
--docVQA_image_dir
--docVQA_ann_path
# DocVQATest
--docVQATest_image_dir
--docVQATest_ann_path

# whether to eval on certain task
--eval_textVQA
--eval_docVQA
--eval_docVQATest
--eval_all

# model name and model path
--model_name
--model_path
# load model from ckpt
--ckpt
# the way the model processes input data, "interleave" represents interleaved image-text form, while "old" represents non-interleaved.
--generate_method

--batchsize

# path to save the outputs
--answer_path
```
<br />

While evaluating on different tasks, parameters need to be set as follows:
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

For the DocVQATest task, in order to upload the inference results to the [official website](https://rrc.cvc.uab.es/?ch=17) for evaluation, run `shell/run_transform.sh` for format transformation after inference. `input_file_path` represents the path to the original output json, `output_file_path` represents the path to the transformed json:
```bash
chmod +x ./shell/run_transform.sh
./shell/run_transform.sh
```

</details>