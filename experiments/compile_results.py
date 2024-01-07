from pathlib import Path
import pandas as pd
import json

def load_results(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return pd.json_normalize(data)

def format_dataframe(df):
    df.columns = [col.replace('test_', '').capitalize() for col in df.columns]
    df.rename(columns={'Rougel': 'RougeL', 'Bleu': 'BLEU', 'Meteor': 'METEOR'}, inplace=True)
    df.replace(replace_dict, inplace=True)
    return df

def process_metrics(df, metrics):
    for metric in metrics:
        df[metric] = df[metric].apply(lambda x: "{:,.2f}".format(float(x)*100))
    return df

finetune_path = Path('../stratch/bounllm/finetuned-models') 
all_results = []

replace_dict = {
    'ul2tr': 'TURNA', 'mt5-large': 'mT5', 'mbart': 'mBART', 
    'mlsum': 'MLSUM', 'tr': 'TRNews', 'tatoeba': 'Tatoeba', 'opensubtitles': 'Opensubtitles'
}
metrics = ['Rouge1', 'Rouge2', 'RougeL', 'BLEU', 'METEOR']

for model_dir in filter(Path.is_dir, finetune_path.iterdir()):
    for task_dir in filter(Path.is_dir, model_dir.iterdir()):
        for result_file in filter(Path.is_file, task_dir.glob('*/results.json')):
            df = load_results(result_file)
            df = df.assign(experiment=result_file.parent.name, 
                           dataset=result_file.parent.name.split('_')[0],
                           mode='_'.join(result_file.parent.name.split('_')[1:]),
                           task=task_dir.name, model=model_dir.name)
            all_results.append(df)

results_df = pd.concat(all_results, ignore_index=True)
results_df = format_dataframe(results_df)

tasks = results_df['Task'].unique()
columns = ['Dataset', 'Model', 'Mode'] + metrics

for task in tasks:
    task_df = results_df[results_df['Task'] == task]
    task_df = process_metrics(task_df.copy(), metrics)
    sorted_task_df = task_df.sort_values(['Dataset', 'Model', 'Mode'])
    if sorted_task_df['Mode'].nunique() == 1: 
        columns.remove("Mode")

    csv_path = f'../results/{task}.csv'
    tex_path = f'../results/{task}.tex'
    sorted_task_df[columns].to_csv(csv_path)
    with open(tex_path, 'w') as f:
        f.write(sorted_task_df[columns].style.hide(axis=0).to_latex())
