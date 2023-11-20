#!/bin/bash
#SBATCH --job-name=ft-summ
#SBATCH --mail-type=ALL
#SBATCH --output=%j.log
#SBATCH --mail-user=gokceuludogan@gmail.com
#SBATCH --container-image ghcr.io\#bouncmpe/cuda-python3
#SBATCH --time=7-00:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G

source /opt/python3/venv/base/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/cu118
cd ~/t5-tuner
pip install -r requirements.txt
pip install wandb
python src/finetune.py --config-name title_generation
#python src/finetune.py --config-name summarization
