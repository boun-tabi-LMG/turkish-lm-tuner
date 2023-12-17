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

echo $1

source /opt/python3/venv/base/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install wandb
cd ~/turkish-lm-tuner
pip install -e . 

python experiments/eval.py --config-name $1
