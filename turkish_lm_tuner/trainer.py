from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    Trainer, Seq2SeqTrainer, 
    TrainingArguments, Seq2SeqTrainingArguments,
    EarlyStoppingCallback

)
from transformers.optimization import Adafactor, AdafactorSchedule

from .evaluator import (
    EvaluatorForClassification,
    EvaluatorForConditionalGeneration
)
import json 
import os


import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
formatter = logging.Formatter('%(levelname)s - %(asctime)s - %(name)s: %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

class BaseModelTrainer:
    def __init__(self, model_name, adafactor_scheduler, training_params):
        self.model_name = model_name
        self.adafactor_scheduler = adafactor_scheduler
        self.training_params = training_params
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def initialize_model(self):
        raise NotImplementedError

    def create_optimizer(self, model):
        logger.info("Creating optimizer")
        if self.adafactor_scheduler:
            logger.info("Using Adafactor with scheduler")
            optimizer = Adafactor(
                model.parameters(),
                scale_parameter=True,
                relative_step=True,
                warmup_init=True,
                lr=None
            )
            lr_scheduler = AdafactorSchedule(optimizer)
        else:
            logger.info("Using Adafactor without scheduler")
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
                warmup_init=False
            )
            lr_scheduler = None
        return optimizer, lr_scheduler


class TrainerForConditionalGeneration(BaseModelTrainer):
    def __init__(self, model_name, task, adafactor_scheduler, training_params, model_save_path, max_target_length, postprocess_fn):
        super().__init__(model_name, adafactor_scheduler, training_params)
        self.evaluator = EvaluatorForConditionalGeneration(model_save_path, model_name, task, max_target_length, training_params, postprocess_fn=postprocess_fn)

    def initialize_model(self):
        return AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
    
    def train_and_evaluate(self, train_dataset, eval_dataset, test_dataset):
        logger.info("Training in conditional generation mode")

        training_args = Seq2SeqTrainingArguments(
            metric_for_best_model='eval_loss',
            load_best_model_at_end=True,
            greater_is_better=False,
            **self.training_params)
        logger.info("Training arguments: %s", training_args)

        model = self.initialize_model()
        optimizer, lr_scheduler = self.create_optimizer(model)

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.evaluator.compute_metrics,
            optimizers=(optimizer, lr_scheduler),
            callbacks = [EarlyStoppingCallback(early_stopping_patience=3)],

        )

        trainer.train()
        results = trainer.evaluate(test_dataset)
        
        logger.info("Results: %s", results)
        json.dump(results, open(os.path.join(self.training_params['output_dir'], "results.json"), "w"))

        return trainer, model


class TrainerForClassification(BaseModelTrainer):
    def __init__(self, model_name, task, adafactor_scheduler, training_params, model_save_path, num_labels):
        super().__init__(model_name, adafactor_scheduler, training_params)
        self.num_labels = num_labels
        self.evaluator = EvaluatorForClassification(model_save_path, model_name, task, training_params)

    def initialize_model(self):
        return AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_labels)
    
    def train_and_evaluate(self, train_dataset, eval_dataset, test_dataset):
        logger.info("Training in classification mode")

        training_args = TrainingArguments(
            metric_for_best_model='eval_loss',
            load_best_model_at_end=True,
            greater_is_better=False,
            **self.training_params)
        logger.info("Training arguments: %s", training_args)

        model = self.initialize_model()
        optimizer, lr_scheduler = self.create_optimizer(model)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.evaluator.compute_metrics,
            optimizers=(optimizer, lr_scheduler),
            callbacks = [EarlyStoppingCallback(early_stopping_patience=3)],
        )
        trainer.train()
        results = trainer.evaluate(test_dataset)
        
        logger.info("Results: %s", results)
        json.dump(results, open(os.path.join(self.training_params['output_dir'], "results.json"), "w"))
        return trainer, model