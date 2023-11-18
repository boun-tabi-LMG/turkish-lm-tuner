# Turkish-T5-Tuner

This repository provides scripts for fine-tuning T5 models on various Turkish NLP tasks. Additionally, it hosts fine-tuned models ready for inference. The goal is to facilitate the usage and adaptation of T5 models for the Turkish language.

## Installation

Clone the repository and install the required dependencies:

```bash
conda create -n t5-tuner python=3.9
conda activate t5-tuner
pip install torch==1.10.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html
git clone https://github.com/boun-llm/t5-tuner.git
cd t5-tuner
pip install -r requirements.txt
```
## Usage
### Fine-tuning
To fine-tune a model on a specific task, ... 
```
python src/finetune.py --config-name <task>
```
### Inference
To run inference with a fine-tuned model, ...
```
python infer.py --model_name .... 
```

## Available Models
List of fine-tuned models available for inference:
- t5-turkish-text-classification
- t5-turkish-ner
