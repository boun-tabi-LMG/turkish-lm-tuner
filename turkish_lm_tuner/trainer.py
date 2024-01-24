from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForSeq2SeqLM,
    Trainer, Seq2SeqTrainer, 
    TrainingArguments, Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
    AutoConfig
)
from transformers.optimization import Adafactor, AdafactorSchedule
from transformers import DataCollatorForTokenClassification
from .evaluator import (
    EvaluatorForClassification,
    EvaluatorForConditionalGeneration
)
from .t5_classifier import T5ForClassification
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
    def __init__(self, model_name, training_params=None, optimizer_params=None):
        self.model_name = model_name
        self.optimizer_params = optimizer_params
        self.training_params = training_params
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def initialize_model(self):
        raise NotImplementedError
    
    def create_adafactor_optimizer(self, model):
        if self.optimizer_params['scheduler']:
            logger.info("Using Adafactor with scheduler")
            default_params = {
                'scale_parameter': True,
                'relative_step': True,
                'warmup_init': True,
                'lr': None
            }
            optimizer = Adafactor(model.parameters(), **default_params)
            lr_scheduler = AdafactorSchedule(optimizer)
        else:
            logger.info("Using Adafactor without scheduler")
            default_params = {
                'lr': 1e-3,
                'eps': (1e-30, 1e-3),
                'clip_threshold': 1.0,
                'decay_rate': -0.8,
                'beta1': None,
                'weight_decay': 0.0,
                'relative_step': False,
                'scale_parameter': False,
                'warmup_init': False
            }
            optimizer = Adafactor(model.parameters(), **default_params)
            lr_scheduler = None
        return optimizer, lr_scheduler
    
    def create_optimizer(self, model):
        logger.info("Creating optimizer")
        optimizer_type = self.optimizer_params['optimizer_type'].lower()        
        if optimizer_type == 'adafactor':
            return self.create_adafactor_optimizer(model)
        else:
            logger.info("Optimizer and scheduler not specified. Continuing with the default parameters.")
            return (None, None)


class TrainerForConditionalGeneration(BaseModelTrainer):
    def __init__(self, model_name, task, training_params, optimizer_params, model_save_path, max_input_length, max_target_length, postprocess_fn):
        super().__init__(model_name, training_params, optimizer_params)
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.evaluator = EvaluatorForConditionalGeneration(model_save_path, model_name, task, max_input_length, max_target_length, training_params, postprocess_fn=postprocess_fn)

    def initialize_model(self):
        return AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
    
    def train_and_evaluate(self, train_dataset, eval_dataset, test_dataset):
        logger.info("Training in conditional generation mode")

        model = self.initialize_model()

        if self.optimizer_params is not None:
            logger.info("Using optimizers with constant parameters")
            optimizer, lr_scheduler = self.create_optimizer(model)
        else:
            logger.info("Using optimizers created based on training_arguments")
            optimizer, lr_scheduler = (None, None)

        generation_config = model.generation_config 
        generation_config.max_length = self.max_input_length
        generation_config.max_new_tokens = self.max_target_length

        logger.info("Generation config: %s", generation_config)

        training_args = Seq2SeqTrainingArguments(
            metric_for_best_model='eval_loss',
            load_best_model_at_end=True,
            greater_is_better=False,
            generation_config=generation_config,
            **self.training_params)
        logger.info("Training arguments: %s", training_args)

        # make datasets smaller for debugging
        # train_dataset = train_dataset.select(range(3))
        # eval_dataset = eval_dataset.select(range(3))

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.evaluator.compute_metrics,
            optimizers=(optimizer, lr_scheduler),
            callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
        )

        trainer.train()
        results = trainer.evaluate(test_dataset)
        
        logger.info("Results: %s", results)
        json.dump(results, open(os.path.join(self.training_params['output_dir'], "results.json"), "w"))

        return trainer, model


class TrainerForClassification(BaseModelTrainer):
    def __init__(self, model_name, task, training_params, optimizer_params, model_save_path, num_labels, postprocess_fn=None):
        super().__init__(model_name, training_params, optimizer_params)
        self.num_labels = num_labels
        self.task = task
        self.evaluator = EvaluatorForClassification(model_save_path, model_name, task, training_params, postprocess_fn=postprocess_fn)

    def initialize_model(self):
        config = AutoConfig.from_pretrained(self.model_name)
        if config.model_type in ["t5", "mt5"]:
            if self.task == "classification":
                return T5ForClassification(self.model_name, config, self.num_labels, "single_label_classification")
            elif self.task in ["ner", "pos_tagging"]:
                return T5ForClassification(self.model_name, config, self.num_labels, "token_classification")
            else:
                return T5ForClassification(self.model_name, config, 1, "regression")
        else:
            if self.task == "classification":
                return AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_labels)
            elif self.task in ["ner", "pos_tagging"]:
                return AutoModelForTokenClassification.from_pretrained(self.model_name, num_labels=self.num_labels)
    
    def train_and_evaluate(self, train_dataset, eval_dataset, test_dataset):
        logger.info("Training in classification mode")

        if self.task in ['ner', 'pos_tagging']:
            data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
            tokenizer = self.tokenizer
        else:
            data_collator, tokenizer = None, None
        training_args = TrainingArguments(
            metric_for_best_model='eval_loss',
            load_best_model_at_end=True,
            greater_is_better=False,
            **self.training_params)
        logger.info("Training arguments: %s", training_args)

        model = self.initialize_model()
        if self.optimizer_params is not None:
            logger.info("Using optimizers with constant parameters")
            optimizer, lr_scheduler = self.create_optimizer(model)
        else:
            logger.info("Using optimizers created based on training_arguments")
            optimizer, lr_scheduler = (None, None)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.evaluator.compute_metrics,
            optimizers=(optimizer, lr_scheduler),
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
        )
        trainer.train()
        results = trainer.evaluate(test_dataset)
        
        logger.info("Results: %s", results)
        json.dump(results, open(os.path.join(self.training_params['output_dir'], "results.json"), "w"))
        return trainer, model
