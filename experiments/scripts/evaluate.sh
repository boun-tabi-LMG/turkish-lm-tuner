#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --mail-type=ALL
#SBATCH --output=outs/%j.log
#SBATCH --mail-user=username@gmail.com
#SBATCH --container-image ghcr.io\#bouncmpe/cuda-python3
#SBATCH --container-mounts /stratch/bounllm/:/stratch/bounllm/
#SBATCH --time=7-00:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G

source /opt/python3/venv/base/bin/activate


pip install turkish-lm-tuner

declare -A tokenizer_mapping=(
    ['turna']='boun-tabi-LMG/TURNA'
    ['mbart']='facebook/mbart-large-cc25'
    ['mt5-large']='google/mt5-large'
)

BASE_PATH=/stratch/bounllm/finetuned-models
TASK_NAME=$1

# Function to run the evaluation
run_evaluation() {
    local model_name=$1
    local dataset_name=$2
    local tokenizer_path=${tokenizer_mapping[$model_name]}

    if [ $TASK_NAME == "ner" ] || [ $TASK_NAME == "pos" ] || [ $TASK_NAME == "qa" ]; then
        CONFIG_NAME=$TASK_NAME"_"$dataset_name
    fi

    echo 'running' $CONFIG_NAME
    python experiments/eval.py --config-name $CONFIG_NAME \
        dataset_name=$dataset_name \
        model_path=$BASE_PATH/$model_name/$TASK_NAME/$dataset_name \
        test_params.output_dir=$BASE_PATH/$model_name/$TASK_NAME/$dataset_name \
        tokenizer_path=$tokenizer_path
    echo $CONFIG_NAME 'done'
}

models=("ul2tr" "mt5-large" "mbart")
if [ $TASK_NAME == "paraphrasing" ]; then
    datasets=("tatoeba" "opensubtitles")
elif [ $TASK_NAME == "ner" ]; then
    datasets=("wikiann" "milliyet")
elif [ $TASK_NAME == "pos" ]; then
    datasets=("boun" "imst")
elif [ $TASK_NAME == "qa" ]; then
    datasets=("exams" "mkqa" "tquad")
fi

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        run_evaluation $model $dataset
    done
done
