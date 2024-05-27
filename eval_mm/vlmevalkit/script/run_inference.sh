export PATH=/usr/local/cuda/bin:$PATH

export HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=1
export timestamp=`date +"%Y%m%d%H%M%S"`
export OLD_VERSION='False'
export PYTHONPATH=$(dirname $SELF_DIR):$PYTHONPATH

# gpu consumed
# fp16 17-18G
# int4 7-8G

# model to be used
# Example: MODELNAME=MiniCPM-Llama3-V-2_5
MODELNAME=$1
# datasets to be tested
# Example: DATALIST="POPE ScienceQA_TEST ChartQA_TEST"
DATALIST=$2
# test mode, all or infer
MODE=$3

echo "Starting inference with model $MODELNAME on datasets $DATALIST"
# run on multi gpus with torchrun command
# remember to run twice, the first run may fail
torchrun --nproc_per_node=8 run.py --data $DATALIST --model $MODELNAME --mode $MODE
torchrun --nproc_per_node=8 run.py --data $DATALIST --model $MODELNAME --mode $MODE
# run on single gpu with python command
# python run.py --data $DATALIST --model $MODELNAME --verbose --mode $MODE
# python run.py --data $DATALIST --model $MODELNAME --verbose --mode $MODE

ls
