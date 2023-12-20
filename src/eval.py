from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    Trainer, TrainingArguments,
    GenerationConfig
)

from omegaconf import DictConfig
from dataset_processor import DatasetProcessor
from metrics import load_task_metrics
import pandas as pd
import hydra
import numpy as np
import os
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
formatter = logging.Formatter('%(levelname)s - %(asctime)s - %(name)s: %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

class BaseEvaluator:
    def __init__(self, model_path, tokenizer_path, task, test_params, postprocess_fn=None):
        self.model_path = model_path
        self.task = task 
        self.test_params = test_params
        self.postprocess_fn = postprocess_fn
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.metrics = load_task_metrics(task)

    def initialize_model(self):
        raise NotImplementedError
    
    def initialize_trainer(self, model):
        test_args = TrainingArguments(
            **self.test_params)
        
        trainer = Trainer(
            model=model,
            args=test_args
        )

        return trainer

    def evaluate_model(self, test_dataset, model=None):
        if not model:
            logger.info("Loading model from %s", self.model_path)
            model = self.initialize_model()

        logger.info("Loading trainer")
        trainer = self.initialize_trainer(model)

        logger.info("Predicting")
        results = trainer.predict(test_dataset)
        return results
    
    def compute_metrics(self, preds, labels):
        scores = {}
        for metric in self.metrics:
            try:
                metric_scores = metric.compute(preds, labels)
            except: # for multi-class classification tasks
                metric_scores = metric.compute(preds, labels, average="weighted")
            scores.update(metric_scores)
        return scores

class EvaluatorForClassification(BaseEvaluator):

    def initialize_model(self):
        # If used without fine-tuning model should be loaded from the model save path
        return AutoModelForSequenceClassification.from_pretrained(self.model_save_path)

    def initialize_trainer(self, model):
        test_args = TrainingArguments(
            **self.test_params)
        
        trainer = Trainer(
            model=model,
            args=test_args,
            compute_metrics=self.compute_metrics,
        )
        return trainer

    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds
        preds = np.argmax(preds[0], axis=-1)

        logger.info("Computing metrics")

        result = super().compute_metrics(preds, labels)

        logger.info("Predictions: %s", preds[:5])
        logger.info("Labels: %s", labels[:5])

        predictions = pd.DataFrame(
            {'Prediction': preds,
             'Label': labels
            })
        
        predictions.to_csv(os.path.join(self.test_params['output_dir'], 'predictions.csv'), index=False)

        logger.info("Result: %s", result)

        return result
    

class EvaluatorForConditionalGeneration(BaseEvaluator):
    def __init__(self, model_path, tokenizer_path, task, max_target_length, test_params, generation_params=None, postprocess_fn=None): 
        super().__init__(model_path, tokenizer_path, task, test_params, postprocess_fn)
        self.max_target_length = max_target_length 
        self.generation_params = generation_params

    def initialize_model(self):
        # If used without fine-tuning model should be loaded from the model save path
        return AutoModelForSeq2SeqLM.from_pretrained(self.model_path)

    def initialize_trainer(self, model): 
        # Set default model parameters
        generation_config = model.generation_config 
        generation_config.max_new_tokens = self.max_target_length

        if self.generation_params: 
            generation_config.update(**self.generation_params)

        logger.info("Generation config: %s", generation_config)
            
        #generation_config = GenerationConfig(**self.generation_params)
        test_args = Seq2SeqTrainingArguments(
            generation_config=generation_config,
            **self.test_params)
        
        trainer = Seq2SeqTrainer(
            model=model,
            args=test_args,
            compute_metrics=self.compute_metrics,
        )
        return trainer
  
    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # Replace -100s used for padding as we can't decode them
        preds = np.where(preds != -100, preds, self.tokenizer.pad_token_id)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        logger.info("Postprocessing predictions and labels")

        # Get post-processing function for specific dataset and task
        processed_preds = self.postprocess_fn(decoded_preds) 
        processed_labels = self.postprocess_fn(decoded_labels)

        predictions = pd.DataFrame(
            {'DecodedPrediction': decoded_preds,
             'DecodedLabel': decoded_labels,
             'Prediction': processed_preds, 
             'Label': processed_labels})
        
        predictions.to_csv(os.path.join(self.test_params['output_dir'], 'predictions.csv'), index=False)


        logger.info("Computing metrics")
        logger.info("Decoded predictions: %s", processed_preds[:5])
        logger.info("Decoded labels: %s", processed_labels[:5])

        result = super().compute_metrics(processed_preds, processed_labels)

        logger.info("Result: %s", result)      

        return result


@hydra.main(config_path="../generation_conf", config_name="default")
def main(cfg: DictConfig):
    model_path = cfg.model_path
    tokenizer_path = cfg.tokenizer_path
    dataset_name = cfg.dataset_name
    task = cfg.task
    task_format = cfg.task_format
    task_mode = cfg.task_mode
    max_input_length = cfg.max_input_length
    max_target_length = cfg.max_target_length
    test_params = cfg.test_params
    generation_params = cfg.generation_params
    dataset_location = cfg.dataset_loc

    logger.info("Loading test dataset")
    dataset_processor = DatasetProcessor(dataset_name, task, task_format, task_mode, model_path, max_input_length, max_target_length, dataset_location)
    test_dataset = dataset_processor.load_and_preprocess_data(split="test")  # Use split="test[:10]" to test for small sample
    postprocess_fn = dataset_processor.dataset.postprocess_data

    logger.info("test_dataset[0]: %s", test_dataset[0])
    logger.info("test_dataset: %s", test_dataset)

    if task_format == 'conditional_generation':
        logger.info("Evaluating in conditional generation mode")
        evaluator = EvaluatorForConditionalGeneration(model_path, tokenizer_path, task, max_target_length, test_params, generation_params, postprocess_fn)
    elif task_format == 'classification':
        logger.info("Evaluating in classification mode")
        evaluator = EvaluatorForClassification(model_path, tokenizer_path, task, test_params)

  
    logger.info("Evaluating model")
    results = evaluator.evaluate_model(test_dataset)
    logger.info("Result: %s", results)    
    json.dump(results.metrics, open(os.path.join(test_params['output_dir'], "test_results.json"), "w"))


if __name__ == "__main__":
    main()
