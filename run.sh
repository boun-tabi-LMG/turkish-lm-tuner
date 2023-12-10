#!/bin/bash
#SBATCH --job-name=sft
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

pip install torch --index-url https://download.pytorch.org/whl/cu118
cd ~/t5-tuner
pip install -r requirements.txt
pip install wandb
# For Single-GPU Training
python src/finetune.py --config-name $1
# For Multi-GPU Training
# python -m torch.distributed.launch --nproc_per_node 3 --use_env src/finetune.py --config-name $1
# Baseline Training
# python src/finetune.py --config-name summarization model_name=google/mt5-large training_params.output_dir=<output_dir>
