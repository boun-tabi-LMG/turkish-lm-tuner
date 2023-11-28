from transformers import (
    PreTrainedTokenizerFast, T5ForConditionalGeneration,
    T5ForSequenceClassification, 
    Trainer, TrainingArguments
)

from transformers.optimization import Adafactor, AdafactorSchedule

import datasets
import hydra
from omegaconf import DictConfig
from dataset import DatasetProcessor
import hydra
import os
local_rank = int(os.environ["LOCAL_RANK"])




class ModelTrainer:
    def __init__(self, model_name, task_format, adafactor_scheduler, training_params):
        self.model_name = model_name
        self.task_format = task_format        
        self.adafactor_scheduler = adafactor_scheduler
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
        
        if self.adafactor_scheduler == True:
            optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
            lr_scheduler = AdafactorSchedule(optimizer)
        else:
            optimizer = Adafactor(
                model.parameters(),
                lr=1e-3,
                eps=(1e-30, 1e-3),
                clip_threshold=1.0,
                decay_rate=-0.8,
                beta1=None,
                weight_decay=0.0,
                relative_step=False,
                scale_parameter=False,
                warmup_init=False,
            )
            lr_scheduler = None
        

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

@hydra.main(config_path="../conf", config_name="default")
def main(cfg: DictConfig):
    model_name = cfg.model_name
    dataset_name = cfg.dataset_name
    task = cfg.task
    task_format = cfg.task_format
    task_mode = cfg.task_mode
    max_input_length = cfg.max_input_length
    max_target_length = cfg.max_target_length
    adafactor_scheduler = cfg.adafactor_scheduler
    training_params = cfg.training_params
    dataset_processor = DatasetProcessor(dataset_name, task, task_format, task_mode, model_name, max_input_length, max_target_length)
    train_set = dataset_processor.load_and_preprocess_data()
    
    try: 
        eval_dataset = dataset_processor.load_and_preprocess_data(split='validation')
        train_dataset = train_set["train"]
    except:
        train_set = train_set.train_test_split(test_size=0.1)
        train_dataset, eval_dataset = train_set["train"], train_set["test"]
    
    test_dataset = dataset_processor.load_and_preprocess_data(split="test")

    model_trainer = ModelTrainer(model_name, task_format, adafactor_scheduler, training_params)
    trainer, model = model_trainer.train_and_evaluate(train_dataset, eval_dataset, test_dataset)

    # model.save_pretrained(model_save_path)
    # dataset_processor.tokenizer.save_pretrained(model_save_path)

if __name__ == "__main__":
    main()
