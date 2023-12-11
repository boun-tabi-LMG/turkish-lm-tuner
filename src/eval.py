from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    Trainer, TrainingArguments,
)

from omegaconf import DictConfig
from dataset import DatasetProcessor
import hydra
import evaluate
import numpy as np
import os
from utils import (
    postprocess_text,
    postprocess_nli,
    postprocess_sts
)

# local_rank = int(os.environ["LOCAL_RANK"])
# TODO: Check their arguments
bleu = evaluate.load("bleu")        # https://huggingface.co/spaces/evaluate-metric/bleu
meteor = evaluate.load("meteor")    # https://huggingface.co/spaces/evaluate-metric/meteor
rouge = evaluate.load("rouge")      # https://huggingface.co/spaces/evaluate-metric/rouge
ter = evaluate.load("ter")          # https://huggingface.co/spaces/evaluate-metric/ter
accuracy = evaluate.load("accuracy")
pearsonr = evaluate.load("pearsonr")

class BaseEvaluator:
    def __init__(self, model_save_path, tokenizer_path, dataset_name, task, test_params):
        self.model_save_path = model_save_path
        self.tokenizer_path = tokenizer_path
        self.dataset_name = dataset_name
        self.task = task 
        self.test_params = test_params
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

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
            model = self.initialize_model()

        trainer = self.initialize_trainer(model)
        results = trainer.predict(test_dataset)
        return trainer, model

class EvaluatorForClassification(BaseEvaluator):
    def __init__(self, model_save_path, tokenizer_path, dataset_name, task, test_params):
        super().__init__(model_save_path, tokenizer_path, dataset_name, task, test_params)

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
        print(preds)
        print(labels)
        
        result = accuracy.compute(predictions=preds, references=labels)
        print(result)
        return result
    

class EvaluatorForConditionalGeneration(BaseEvaluator):
    def __init__(self, model_save_path, tokenizer_path, dataset_name, task, max_target_length, test_params): # generation_params
        super().__init__(model_save_path, tokenizer_path, dataset_name, task, test_params)
        self.max_target_length = max_target_length 
        #self.generation_params = generation_params

    def initialize_model(self):
        # If used without fine-tuning model should be loaded from the model save path
        return AutoModelForSeq2SeqLM.from_pretrained(self.model_save_path)

    def initialize_trainer(self, model):
        generation_config = model.generation_config 
        generation_config.max_new_tokens = self.max_target_length

        test_args = Seq2SeqTrainingArguments(
            generation_config=generation_config,
            **self.test_params)
        
        trainer = Seq2SeqTrainer(
            model=model,
            args=test_args,
            compute_metrics=self.compute_metrics,
        )
        return trainer
  
    def get_postprocess_function(self):
        # Mapping of dataset_name and task to corresponding postprocess functions
        postprocess_functions = {
            ('exams', 'question_answering'): postprocess_text,
            ('exams', 'question_generation'): postprocess_text,
            ("xquad", "question_answering"): postprocess_text,
            ("xquad", "question_generation"): postprocess_text,
            ("mkqa", "question_answering"): postprocess_text,
            ("mkqa", "question_generation"): postprocess_text,
            ("wikiann", "ner"): postprocess_text,
            ("xtreme", "ner"): postprocess_text,
            ("stsb_tr", "semantic_similarity") : postprocess_sts,
            ("nli_tr", "nli") : postprocess_nli,
            ("snli_tr", "nli") : postprocess_nli,
            ("multinli_tr", "nli") : postprocess_nli,
            # ... add mappings for other dataset and task type combinations
        }
        return postprocess_functions.get((self.dataset_name, self.task), postprocess_text)
    
    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # Replace -100s used for padding as we can't decode them
        preds = np.where(preds != -100, preds, self.tokenizer.pad_token_id)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)


        # Get post-processing function for specific dataset and task
        postprocess_function = self.get_postprocess_function()
        decoded_preds, decoded_labels = postprocess_function(decoded_preds, decoded_labels)

        if self.task in ['summarization', "paraphrasing", "title_generation"]:
            result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
            result = {key: value * 100 for key, value in result.items()}
            prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]
            result["gen_len"] = np.mean(prediction_lens)
            result = {k: round(v, 4) for k, v in result.items()}
        elif self.task == "nli":
            result = accuracy.compute(predictions=decoded_preds, references=decoded_labels)
        elif self.task == "semantic_similarity":
            result = pearsonr.compute(predictions=decoded_preds, references=decoded_labels)
        if self.task == 'paraphrasing':
            meteor_score = meteor.compute(predictions=decoded_preds, references=decoded_labels)
            bleu_score = bleu.compute(predictions=decoded_preds, references=decoded_labels)
            ter_score = ter.compute(predictions=decoded_preds, references=decoded_labels, case_insensitive=True)
            result = {**result, **meteor_score, **bleu_score, **ter_score}        
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
    if task_format == 'conditional_generation':
        print("******Conditional Generation Mode******")
        evaluator = EvaluatorForConditionalGeneration(model_path, model_name, dataset_name, task, max_target_length, test_params)
    elif task_format == 'classification':
        print("******Classification Mode******")
        evaluator = EvaluatorForClassification(model_path, model_name, dataset_name, task, test_params)

    dataset_processor = DatasetProcessor(dataset_name, task, task_format, task_mode, model_name, max_input_length, max_target_length, dataset_location)
    test_dataset = dataset_processor.load_and_preprocess_data(split="test")  # Use split="test[:10]" to test for small sample

    print(test_dataset[0])
    print("test", test_dataset)

    evaluator.evaluate_model(test_dataset)


if __name__ == "__main__":
    main()
