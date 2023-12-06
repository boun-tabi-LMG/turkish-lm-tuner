from transformers import (
    PreTrainedTokenizerFast, T5ForConditionalGeneration,
    T5ForSequenceClassification, 
    Seq2SeqTrainer, TrainingArguments, Seq2SeqTrainingArguments,
    EarlyStoppingCallback
)

from transformers.optimization import Adafactor, AdafactorSchedule

from omegaconf import DictConfig
from utils import postprocess_text
from dataset import DatasetProcessor
from eval import Evaluator
import hydra
import evaluate
import numpy as np
import os
# local_rank = int(os.environ["LOCAL_RANK"])

rouge = evaluate.load("rouge")



class ModelTrainer:
    def __init__(self, model_name, task, task_format, adafactor_scheduler, training_params):
        self.model_name = model_name
        self.task = task
        self.task_format = task_format        
        self.adafactor_scheduler = adafactor_scheduler
        self.training_params = training_params
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)

    def initialize_model(self):
        if self.task_format == "conditional_generation":
            model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        elif self.task_format == "classification":
            model = T5ForSequenceClassification.from_pretrained(self.model_name)
        return model

    def initialize_evaluator(self, model_save_path, dataset_name, max_target_length):
        self.evaluator = Evaluator(model_save_path, self.model_name, dataset_name, self.task, self.task_format, max_target_length, self.training_params)

    def train_and_evaluate(self, train_dataset, eval_dataset, test_dataset):
        # TODO: Should we change this with Seq2SeqTrainingArguments?
        # TODO: predict_with_generate, generation_max_length, generation_num_beams, generation_config
        training_args = Seq2SeqTrainingArguments(
            report_to="wandb",
            metric_for_best_model='eval_loss',
            load_best_model_at_end=True,
            greater_is_better=False,
            **self.training_params)
        
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
        

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset, 
            compute_metrics = self.evaluator.compute_metrics,
            optimizers=(optimizer, lr_scheduler),
            callbacks = [EarlyStoppingCallback(early_stopping_patience=3)],

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
    dataset_location = cfg.dataset_loc
    dataset_processor = DatasetProcessor(dataset_name, task, task_format, task_mode, model_name, max_input_length, max_target_length, dataset_location)
    train_set = dataset_processor.load_and_preprocess_data()
    model_save_path = training_params['output_dir']
    
    try: 
        eval_dataset = dataset_processor.load_and_preprocess_data(split='validation')
        train_dataset = train_set
    except:
        train_set = train_set.train_test_split(test_size=0.1)
        train_dataset, eval_dataset = train_set["train"], train_set["test"]
    
    test_dataset = dataset_processor.load_and_preprocess_data(split="test")

    print("train", train_dataset)
    print("val", eval_dataset)
    print("test", test_dataset)

    model_trainer = ModelTrainer(model_name, task, task_format, adafactor_scheduler, training_params)
    model_trainer.initialize_evaluator(model_save_path, dataset_name, max_target_length)
    trainer, model = model_trainer.train_and_evaluate(train_dataset, eval_dataset, test_dataset)

    print("Best model saved at", model_save_path)
    model.save_pretrained(model_save_path)
    # dataset_processor.tokenizer.save_pretrained(model_save_path)

if __name__ == "__main__":
    main()
