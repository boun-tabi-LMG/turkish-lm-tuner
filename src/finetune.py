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
            compute_metrics = self.compute_metrics,
            optimizers=(optimizer, lr_scheduler),
            callbacks = [EarlyStoppingCallback(early_stopping_patience=3)],

        )

        trainer.train()
        results = trainer.evaluate(test_dataset)
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

        return result


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

    model_trainer = ModelTrainer(model_name, task, task_format, adafactor_scheduler, training_params)
    trainer, model = model_trainer.train_and_evaluate(train_dataset, eval_dataset, test_dataset)

    # model.save_pretrained(model_save_path)
    # dataset_processor.tokenizer.save_pretrained(model_save_path)

if __name__ == "__main__":
    main()
