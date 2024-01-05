import os
from transformers import AutoModel, AutoTokenizer
from pathlib import Path 
from huggingface_hub import create_repo

HF_AUTH_TOKEN = os.getenv('HF_AUTH_TOKEN')

tokenizer_mapping = {
    'ul2tr': '/stratch/bounllm/pretrained_checkpoints/ckpt-1.74M/',
    'mbart': 'facebook/mbart-large-cc25', 
    'mt5-large': 'google/mt5-large'
}

completed_tasks = ['summarization', 'paraphrasing', 'title_generation', 'semantic_similarity', 'nli']
path = Path('/stratch/bounllm/finetuned-models/')
organization = "boun-tabi-LMG"
for model_dir in path.iterdir():
    if model_dir.name not in tokenizer_mapping:
        continue

    for task in model_dir.iterdir():
        if task.name not in completed_tasks:
            continue

        for dataset in task.iterdir():
            model_name = 'turna' if model_dir.name == 'ul2tr' else model_dir.name
            repo = f'{organization}/{model_name}_{task.name}_{dataset.name}'
            try: 
                repo_url = create_repo(repo, private=True, token=HF_AUTH_TOKEN)
                print(f'{repo_url} has been successfully created.')
                model = AutoModel.from_pretrained(dataset)
                model.push_to_hub(repo, private=True, token=HF_AUTH_TOKEN)
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_mapping[model_dir.name])
                tokenizer.push_to_hub(repo, private=True, token=HF_AUTH_TOKEN)

            except Exception as e:
                print(f'Error during pushing {dataset}. Either repo exists or a problem occurred during uploding files, check error: {e}')