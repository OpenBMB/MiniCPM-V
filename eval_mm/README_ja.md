# 評価

## opencompass
まず、`vlmevalkit`ディレクトリに入り、すべての依存関係をインストールします：
```bash
cd vlmevalkit
pip install -r requirements.txt
```
<br />

次に、`script/run_inference.sh`を実行します。これは、`MODELNAME`、`DATALIST`、および`MODE`の3つの入力パラメータを順に受け取ります。`MODELNAME`はモデルの名前、`DATALIST`は推論に使用するデータセット、`MODE`は評価モードを表します：
```bash
chmod +x ./script/run_inference.sh
./script/run_inference.sh $MODELNAME $DATALIST $MODE
```
<br />

`MODELNAME`の3つの選択肢は`vlmeval/config.py`にリストされています：
```bash
ungrouped = {
    'MiniCPM-V':partial(MiniCPM_V, model_path='openbmb/MiniCPM-V'),
    'MiniCPM-V-2':partial(MiniCPM_V, model_path='openbmb/MiniCPM-V-2'),
    'MiniCPM-Llama3-V-2_5':partial(MiniCPM_Llama3_V, model_path='openbmb/MiniCPM-Llama3-V-2_5'),
}
```
<br />

すべての利用可能な`DATALIST`の選択肢は`vlmeval/utils/dataset_config.py`にリストされています。単一のデータセットで評価する場合、データセット名を引用符なしで直接呼び出します。複数のデータセットで評価する場合、異なるデータセット名をスペースで区切り、両端に引用符を追加します：
```bash
$DATALIST="POPE ScienceQA_TEST ChartQA_TEST"
```
<br />

各ベンチマークで直接スコアリングする場合、`MODE=all`を設定します。推論結果のみが必要な場合、`MODE=infer`を設定します。ホームページに表示されている表の結果（MMEからRealWorldQAまでの列）を再現するには、次の設定に従ってスクリプトを実行する必要があります：
```bash
# すべての7つのデータセットで実行
./script/run_inference.sh MiniCPM-Llama3-V-2_5 "MME MMBench_TEST_EN MMBench_TEST_CN MMMU_DEV_VAL MathVista_MINI LLaVABench RealWorldQA" all

# 単一のデータセットで実行するための指示
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
まず、`vqaeval`ディレクトリに入り、すべての依存関係をインストールします。次に、すべてのタスクのデータセットをダウンロードするために`downloads`サブディレクトリを作成します：
```bash
cd vqaeval
pip install -r requirements.txt
mkdir downloads
```
<br />

次のリンクからデータセットをダウンロードし、指定されたディレクトリに配置します：
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
# https://rrc.cvc.uab.es/?ch=17&com=downloads から Task 1 - Single Page Document Visual Question Answering の Images と Annotations をダウンロード
# spdocvqa_images.tar.gz と spdocvqa_qas.zip を DocVQA ディレクトリに移動
tar -zxvf spdocvqa_images.tar.gz -C spdocvqa_images && rm spdocvqa_images.tar.gz
unzip spdocvqa_qas.zip && rm spdocvqa_qas.zip
cp spdocvqa_qas/val_v1.0_withQT.json . && cp spdocvqa_qas/test_v1.0.json .  && rm -rf spdocvqa_qas
cd ../..
```
<br />

`downloads`ディレクトリは次の構造に従って整理されるべきです：
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

`shell/run_inference.sh`のパラメータを変更し、推論を実行します：

```bash
chmod +x ./shell/run_inference.sh
./shell/run_inference.sh
```
<br />

すべてのオプションパラメータは`eval_utils/getargs.py`にリストされています。主要なパラメータの意味は次のとおりです：
```bash
# 画像とそれに対応する質問のパス
# TextVQA
--textVQA_image_dir
--textVQA_ann_path
# DocVQA
--docVQA_image_dir
--docVQA_ann_path
# DocVQATest
--docVQATest_image_dir
--docVQATest_ann_path

# 特定のタスクで評価するかどうか
--eval_textVQA
--eval_docVQA
--eval_docVQATest
--eval_all

# モデル名とモデルパス
--model_name
--model_path
# ckptからモデルをロード
--ckpt
# モデルが入力データを処理する方法、"interleave"は画像とテキストが交互に配置される形式、"old"は非交互形式を表します。
--generate_method

--batchsize

# 出力を保存するパス
--answer_path
```
<br />

異なるタスクで評価する場合、次のようにパラメータを設定する必要があります：
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

DocVQATestタスクの場合、推論結果を[公式サイト](https://rrc.cvc.uab.es/?ch=17)にアップロードして評価するために、推論後に`shell/run_transform.sh`を実行して形式変換を行います。`input_file_path`は元の出力jsonのパスを表し、`output_file_path`は変換後のjsonのパスを表します：
```bash
chmod +x ./shell/run_transform.sh
./shell/run_transform.sh
```
