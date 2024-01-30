from transformers import AutoModelForSeq2SeqLM, AutoConfig, AutoModelForSequenceClassification, AutoModelForTokenClassification
from .dataset_processor import DatasetProcessor
from .t5_classifier import T5ForClassification
from dataclasses import dataclass
import torch

@dataclass
class TaskConfig:
    max_new_tokens: int = None
    length_penalty: float = None
    no_repeat_ngram_size: int = None
    early_stopping: bool = True
    decoder_start_token_id: int = None
    eos_token_id: int = None
    pad_token_id: int = None
    num_beams: int = 1
    repetition_penalty: float = None
    top_p: float = None
    top_k: int = None
    temperature: float = None
    do_sample: bool = None
    max_length: int = None
    num_return_sequences: int = 1

# Generation parameters for each task
task_parameters = {
    'categorization': {
        'max_new_tokens': 8
    },
    'nli': {
        'max_new_tokens': 8
    },
    'sentiment': {
        'max_new_tokens': 4
    },
    'ner': {
        'length_penalty': 2.0,
        'no_repeat_ngram_size': 3,
        'max_new_tokens': 64
    },
    'pos_tagging': {
        'length_penalty': 2.0,
        'no_repeat_ngram_size': 3,
        'max_new_tokens': 64
    },
    'paraphrasing': {
        'max_new_tokens': 20
    },
    'summarization': {
        'length_penalty': 2.0,
        'no_repeat_ngram_size': 3,
        'max_new_tokens': 128,
        'early_stopping': True,
        'num_beams': 4,
        'decoder_start_token_id': 0,
        'eos_token_id': 1,
        'pad_token_id': 0
    },
    'title_generation': {
        'length_penalty': 2.0,
        'no_repeat_ngram_size': 3,
        'max_new_tokens': 128,
        'decoder_start_token_id': 0,
        'eos_token_id': 1,
        'pad_token_id': 0
    },
    'sts': {
        'max_new_tokens': 10
    },
    'generation': {
        'length_penalty': 1.0,
        'no_repeat_ngram_size': 3,
        'max_new_tokens': 128,
        'do_sample': True,
        'num_beams': 3,
        'repetition_penalty': 3.0,
        'top_p': 0.95,
        'top_k': 10,
        'temperature': 1,
        'early_stopping': True,
        'max_length': 256
    }
}


class BasePredictor:
    """
    Base class for all predictors
    Args:
        model_name: Model name or path
        task: Task name
        task_format: Task format. It can be either 'classification' or 'conditional_generation'
        task_mode: Task mode. It can be either '', '[NLU]', '[NLG]' or '[S2S]' depending on the model and task
        max_input_length: Maximum input length
    """
    def __init__(self, model_name, task, task_format='conditional_generation', task_mode='', max_input_length=512):
        self.processor = DatasetProcessor(task=task, task_format=task_format, task_mode=task_mode, tokenizer_name=model_name, max_input_length=max_input_length)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.initialize_model(model_name, task, task_format=task_format)
        self.model.to(self.device)
        self.task_format = task_format

    def initialize_model(self, model_name, task, task_format='conditional_generation'):
        if task_format == "conditional_generation":
            return AutoModelForSeq2SeqLM.from_pretrained(model_name)
        else:
            config = AutoConfig.from_pretrained(model_name)
            if config.model_type in ["t5", "mt5"]:
                return T5ForClassification.from_pretrained(model_name)
            else:
                if task == "classification":
                    return AutoModelForSequenceClassification.from_pretrained(model_name)
                elif task in ["ner", "pos_tagging"]:
                    return AutoModelForTokenClassification.from_pretrained(model_name)

    def predict(self, text, generation_config=None):
        inputs = self.processor.tokenize_function({'input_text': [text]}, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if self.task_format == "conditional_generation":
            outputs = self.model.generate(**inputs, **generation_config)
            return self.processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        else:
            return self.model(**inputs)

class LabelPredictor(BasePredictor):
    """
    Predictor class for classification models
    """
    def __init__(self, model_name, task, task_format='classification', max_input_length=512):
        super().__init__(model_name, task, task_format, max_input_length)

    def predict(self, text):
        return super().predict(text)
    

class TextPredictor(BasePredictor):
    """
    Predictor class for conditional generation models
    """
    def __init__(self, model_name, task, task_format='conditional_generation', max_input_length=512):
        super().__init__(model_name, task, task_format, max_input_length=max_input_length)
        self.task_config = TaskConfig(**task_parameters[task])

    def predict(self, text, **kwargs):
        generation_config = vars(self.task_config, **kwargs) if self.task_format == 'conditional_generation' else {}
        return super().predict(text, generation_config)
    