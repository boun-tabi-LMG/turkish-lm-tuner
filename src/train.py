from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    T5ForSequenceClassification, Trainer,
    TrainingArguments
)
import datasets

dataset_mapping = {
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
    def __init__(self, dataset_name, task_type, tokenizer_name):
        self.dataset_name = dataset_name
        self.task_type = task_type
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

    def load_and_preprocess_data(self, split='train'):
        dataset = datasets.load_dataset(self.dataset_name, split=split)
        preprocess_function = self.get_preprocess_function()
        processed_dataset = dataset.map(preprocess_function, batched=True)
        tokenized_dataset = processed_dataset.map(self.tokenize_function, batched=True)
        return tokenized_dataset
    
    def get_preprocess_function(self, dataset_name, task_type):
        # Mapping of dataset_name and task_type to corresponding preprocess functions
        preprocess_functions = {
            ('trnews', 'summarization'): self.preprocess_trnews_summarization,
            ('trnews', 'title_generation'): self.preprocess_trnews_title_generation,
            ('opensubtitles', 'paraphrasing'): self.preprocess_paraphrasing, 
            ('ted', 'paraphrasing'): self.preprocess_paraphrasing,
            ('tatoeba', 'paraphrasing'): self.preprocess_paraphrasing, 

            # ... add mappings for other dataset and task type combinations
        }
        return preprocess_functions.get((dataset_name, task_type), self.default_preprocess_function)

    def default_preprocess_function(self, examples):
        # Default preprocessing if specific preprocess function is not found
        return {"input_text": examples["text"]}

    def preprocess_trnews_summarization(self, examples):
        return {"input_text": examples["content"], "target_text": examples["abstract"]}

    def preprocess_trnews_title_generation(self, examples): 
        return {"input_text": examples["content"], "target_text": examples["title"]}

    def preprocess_paraphrasing(self, examples): 
        return {"input_text": examples["src"], "target_text": examples["tgt"]}

    def tokenize_function(self, examples):
        if self.task_type == 'conditional_generation':
            return (
                self.tokenizer(
                    examples["input_text"],
                    padding="max_length",
                    truncation=True, 
                ), 
                self.tokenizer(
                    examples["target_text"],
                    padding="max_length",
                    truncation=True, 
                )
            )    

        return self.tokenizer(
            examples["input_text"],
            padding="max_length",
            truncation=True
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

    def train_and_evaluate(self, train_dataset, eval_dataset):
        training_args = TrainingArguments(**self.training_params)
        model = self.initialize_model()
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset["train"],
            eval_dataset=train_dataset["validation"],
        )
        trainer.train()
        results = trainer.evaluate(eval_dataset)
        print(results)
        return trainer, model

def main(model_name, task_name, task_format, training_params, model_save_path):
    dataset_processor = DatasetProcessor(task_name, task_format, model_name)
    train_dataset = dataset_processor.load_and_preprocess_data()
    eval_dataset = dataset_processor.load_and_preprocess_data(split='test')

    model_trainer = ModelTrainer(model_name, task_format, training_params)
    trainer, model = model_trainer.train_and_evaluate(train_dataset, eval_dataset)

    model.save_pretrained(model_save_path)
    dataset_processor.tokenizer.save_pretrained(model_save_path)

# Example usage:
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
    'task_format': 'classification'
}

main('google/mt5-small', 'tr_news', 'summarization', training_params, './t5_finetuned')
