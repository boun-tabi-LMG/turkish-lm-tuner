from tr_datasets import initialize_dataset, DATASET_MAPPING_NAMES

dataset_name = "boun"

dataset = initialize_dataset(dataset_name, dataset_loc=None)

for split in ["train", "validation", "test"]:
    print(dataset.load_dataset(split)[0])

all_splits = dataset.load_dataset()
for split in ["train", "validation", "test"]:
    print(all_splits[split][0])


data = dataset.load_dataset("train")

column_names = data.column_names
column_names = [col for col in column_names if col not in ['input_text', 'target_text', 'label']]
processed_data = data.map(dataset.preprocess_data, remove_columns=column_names, batched=True)
print(processed_data[0])
