from transformers import (
    PreTrainedTokenizerFast, T5ForConditionalGeneration,
    # T5ForSequenceClassification, 
    Trainer, TrainingArguments
)

from transformers.optimization import Adafactor, AdafactorSchedule

import datasets
import argparse

dataset_mapping = {
    "offensive": "Toygar/turkish-offensive-language-detection",
    # summarization/title generation
    "tr_news": "batubayk/TR-News",
    # paraphrasing
    "opensubtitles": "mrbesher/tr-paraphrase-opensubtitles2018",
    "tatoeba": "mrbesher/tr-paraphrase-tatoeba",
    "ted": "mrbesher/tr-paraphrase-ted2013",
    # translation

    # question answering/generation?

    # nli
    "nli_tr": "nli_tr",
    # semantic textual similarity

    # ner

    # pos tagging


    # text classification
}

class DatasetProcessor:
    def __init__(self, dataset_name, task, task_format, tokenizer_name):
        self.dataset_name = dataset_name
        self.task = task
        self.task_format = task_format
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_name)

    def load_and_preprocess_data(self, split='train'):
        dataset = datasets.load_dataset(dataset_mapping[self.dataset_name], split=split) #.select(range(100))
        preprocess_function = self.get_preprocess_function()
        column_names = dataset.column_names
        processed_dataset = dataset.map(preprocess_function, remove_columns=column_names, batched=True)
        tokenized_dataset = processed_dataset.map(self.tokenize_function, batched=True)
        return tokenized_dataset

    def get_preprocess_function(self):
        # Mapping of dataset_name and task to corresponding preprocess functions
        preprocess_functions = {
            ('trnews', 'summarization'): self.preprocess_trnews_summarization,
            ('trnews', 'title_generation'): self.preprocess_trnews_title_generation,
            ('opensubtitles', 'paraphrasing'): self.preprocess_paraphrasing,
            ('ted', 'paraphrasing'): self.preprocess_paraphrasing,
            ('tatoeba', 'paraphrasing'): self.preprocess_paraphrasing,

            # ... add mappings for other dataset and task type combinations
        }
        return preprocess_functions.get((self.dataset_name, self.task), self.default_preprocess_function)

    def default_preprocess_function(self, examples):
        # Default preprocessing if specific preprocess function is not found
        return {"input_text": examples["text"], "labels": examples["label"]}

    def preprocess_trnews_summarization(self, examples):
        return {"input_text": examples["content"], "target_text": examples["abstract"]}

    def preprocess_trnews_title_generation(self, examples):
        return {"input_text": examples["content"], "target_text": examples["title"]}

    def preprocess_paraphrasing(self, examples):
        return {"input_text": examples["src"], "target_text": examples["tgt"]}

    def preprocess_nli(self, examples):
        return {"input_text": examples["premise"] + ' [SEP]' + examples["hypothesis"]}

    def tokenize_function(self, examples):
        if self.task_format == 'conditional_generation':
            inputs_tokenized = self.tokenizer(
                        examples["input_text"],
                        padding="max_length",
                        truncation=True,
                        max_length=128,
                   )
            targets_tokenized = self.tokenizer(
                        examples["target_text"],
                        padding="max_length",
                        truncation=True,
                        max_length=128,
                   )
            return {'labels': targets_tokenized['input_ids'], **inputs_tokenized}

        return self.tokenizer(
            examples["input_text"],
            padding="max_length",
            truncation=True,
            max_length=128,
        )


class ModelTrainer:
    def __init__(self, model_name, task_format, training_params):
        self.model_name = model_name
        self.task_format = task_format
        self.training_params = training_params

    def initialize_model(self):
        if self.task_format == "conditional_generation":
            model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        elif self.task_format == "classification":
            model = T5ForSequenceClassification.from_pretrained(self.model_name)

        return model

    def train_and_evaluate(self, train_dataset, eval_dataset, test_dataset):
        training_args = TrainingArguments(**self.training_params)
        model = self.initialize_model()
        optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
        lr_scheduler = AdafactorSchedule(optimizer)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset, 
            optimizers=(optimizer, lr_scheduler)
        )
        trainer.train()
        results = trainer.evaluate(test_dataset)
        print(results)
        return trainer, model

def main(model_name, dataset_name, task, task_format, training_params): # , model_save_path):
    dataset_processor = DatasetProcessor(dataset_name, task, task_format, model_name)
    train_set = dataset_processor.load_and_preprocess_data()
    train_set = train_set.train_test_split(test_size=0.1)
    train_dataset, eval_dataset = train_set["train"], train_set["test"]
    test_dataset = dataset_processor.load_and_preprocess_data(split="test")

    model_trainer = ModelTrainer(model_name, task_format, training_params)
    trainer, model = model_trainer.train_and_evaluate(train_dataset, eval_dataset, test_dataset)

    # model.save_pretrained(model_save_path)
    # dataset_processor.tokenizer.save_pretrained(model_save_path)
if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser(description='Train a model on a dataset for a specific task.')

    # Adding arguments
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to be trained')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset to be used')
    parser.add_argument('--task', type=str, required=True, help='Task for which the model is being trained')
    parser.add_argument('--task_format', type=str, required=True, help='Format of the task')

    # Parse arguments
    args = parser.parse_args()

    # Example training parameters 
    training_params = {
        'per_device_train_batch_size': 8,
        'per_device_eval_batch_size': 8,
        'num_train_epochs': 3,
        'evaluation_strategy': "epoch",
        'save_strategy': "epoch",
        'logging_dir': './logs',
        'logging_steps': 100,
        'save_total_limit': 3,
        'remove_unused_columns': False,
        'push_to_hub': False,
        'output_dir': './t5_finetuned_test', 
    }
    
    main(args.model_name, args.dataset_name, args.task, args.task_format, training_params)

# main('t5-small', 'offensive', 'classification', 'classification', training_params) 
# main('t5-small', 'ted', 'paraphrasing', 'conditional_generation', training_params)
