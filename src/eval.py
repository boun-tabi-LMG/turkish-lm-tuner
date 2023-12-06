from transformers import (
    PreTrainedTokenizerFast, T5ForConditionalGeneration,
    T5ForSequenceClassification, 
    Seq2SeqTrainer, TrainingArguments, Seq2SeqTrainingArguments,
)

from omegaconf import DictConfig
from utils import postprocess_text
from dataset import DatasetProcessor
import hydra
import evaluate
import numpy as np
import os

# local_rank = int(os.environ["LOCAL_RANK"])

rouge = evaluate.load("rouge")
accuracy = evaluate.load("accuracy")

class Evaluator:
    def __init__(self, model_save_path, tokenizer_path, task, task_format, max_target_length, test_params): # generation_params
        self.model_save_path = model_save_path
        self.tokenizer_path = tokenizer_path
        self.task = task
        self.task_format = task_format      
        self.max_target_length = max_target_length  
        self.test_params = test_params
        #self.generation_params = generation_params
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

    def initialize_model(self):
        if self.task_format == "conditional_generation":
            model = T5ForConditionalGeneration.from_pretrained(self.model_save_path)
        elif self.task_format == "classification":
            model = T5ForSequenceClassification.from_pretrained(self.model_save_path)
        self.model = model
        return model

    def evaluate_model(self, test_dataset):
        
        model = self.initialize_model()
        generation_config = model.generation_config 
        generation_config.max_new_tokens = self.max_target_length

        test_args = Seq2SeqTrainingArguments(
            generation_config = generation_config,
            **self.test_params)
        
        trainer = Seq2SeqTrainer(
            model=model,
            args=test_args,
            #eval_dataset=eval_dataset, 
            compute_metrics = self.compute_metrics,
        )

        results = trainer.predict(test_dataset)
        print(results)
        return trainer, model
    
    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # Replace -100s used for padding as we can't decode them
        preds = np.where(preds != -100, preds, self.tokenizer.pad_token_id)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        if self.task == 'summarization':
            # TODO: Check if rouge is working correctly or need to be fixed
            # decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
            # decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
            # result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True, use_aggregator=True)

            result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
            result = {key: value * 100 for key, value in result.items()}
            prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]
            result["gen_len"] = np.mean(prediction_lens)
            result = {k: round(v, 4) for k, v in result.items()}
        else:
            result = accuracy.compute(predictions=decoded_preds, references=decoded_labels)
            print(result)

        return result


@hydra.main(config_path="../generation_conf", config_name="default")
def main(cfg: DictConfig):
    model_path = cfg.model_path
    model_name = cfg.model_name
    dataset_name = cfg.dataset_name
    task = cfg.task
    task_format = cfg.task_format
    task_mode = cfg.task_mode
    max_input_length = cfg.max_input_length
    max_target_length = cfg.max_target_length
    test_params = cfg.test_params
    dataset_location = cfg.dataset_loc

    evaluator = Evaluator(model_path, model_name, task, task_format, max_target_length, test_params)

    dataset_processor = DatasetProcessor(dataset_name, task, task_format, task_mode, model_name, max_input_length, max_target_length, dataset_location)
    test_dataset = dataset_processor.load_and_preprocess_data(split="test[:10]")

    print(test_dataset[0])
    print("test", test_dataset)

    evaluator.evaluate_model(test_dataset)


if __name__ == "__main__":
    main()
