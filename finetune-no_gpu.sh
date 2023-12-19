#!/bin/bash
#SBATCH --job-name=sft
#SBATCH --mail-type=ALL
#SBATCH --output=outs/%j.log
#SBATCH --mail-user=username@gmail.com
#SBATCH --container-image ghcr.io\#bouncmpe/cuda-python3
#SBATCH --container-mounts /stratch/bounllm/:/stratch/bounllm/
#SBATCH --time=7-00:00:00

echo $1

source /opt/python3/venv/base/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/cu118
cd ~/t5-tuner
pip install -r requirements.txt
pip install wandb
python src/finetune.py --config-name $1
