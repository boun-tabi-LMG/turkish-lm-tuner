import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from pathlib import Path
from huggingface_hub import create_repo

# Set up the Hugging Face authentication token
HF_AUTH_TOKEN = os.getenv('HF_AUTH_TOKEN')

# Tokenizer mapping and model paths
tokenizer_mapping = {
    'ul2tr': '/stratch/bounllm/pretrained_checkpoints/ckpt-1.74M/',
    'mbart': 'facebook/mbart-large-cc25', 
    'mt5-large': 'google/mt5-large'
}

# List of completed tasks
completed_tasks = ['summarization', 'paraphrasing', 'title_generation', 'semantic_similarity', 'nli', 'ner', 'pos']

# Organization name
organization = "boun-tabi-LMG"

# Function to push models and tokenizers to Hugging Face Hub
def push_model_to_hub(repo_name, model_path, tokenizer_path):
    try:
        # delete_repo(repo_name, token=HF_AUTH_TOKEN)
        repo_url = create_repo(repo_name, private=True, token=HF_AUTH_TOKEN)
        print(f'{repo_url} has been successfully created.')
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        model.push_to_hub(repo_name, private=True, token=HF_AUTH_TOKEN)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        tokenizer.push_to_hub(repo_name, private=True, token=HF_AUTH_TOKEN)
    except Exception as e:
        print(f'Error during pushing {model_path}. Repo exists or an error occurred, check error: {e}')

# Push pretrained models
pretrained_models = ['1.74M', '1M', '500K']
for model_version in pretrained_models:
    model_path = f'/stratch/bounllm/pretrained_checkpoints/ckpt-{model_version}'
    repo_name = f'{organization}/turna-{model_version}'
    push_model_to_hub(repo_name, model_path, tokenizer_mapping['ul2tr'])

# Push finetuned models
path = Path('/stratch/bounllm/finetuned-models/')
for model_dir in path.iterdir():
    if model_dir.name not in tokenizer_mapping:
        continue

    for task in model_dir.iterdir():
        if task.name not in completed_tasks:
            continue

        for dataset in task.iterdir():
            model_name = 'turna' if model_dir.name == 'ul2tr' else model_dir.name
            repo_name = f'{organization}/{model_name}_{task.name}_{dataset.name}'
            push_model_to_hub(repo_name, dataset, tokenizer_mapping[model_dir.name])
