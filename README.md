<h1 align="center">  🦖 Turkish LM Tuner </h1>
<h4 align="center"> Summary of project or library comes here. </h4>

</br>

[![PyPI](https://img.shields.io/pypi/v/turkish-lm-tuner)](https://pypi.org/project/turkish-lm-tuner/)
[![Conda](https://img.shields.io/conda/v/conda-forge/turkish-lm-tuner?label=conda&color=success)](https://anaconda.org/conda-forge/turkish-lm-tuner)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/turkish-lm-tuner)](https://pypi.org/project/turkish-lm-tuner/)
[![Conda](https://img.shields.io/conda/dn/conda-forge/turkish-lm-tuner)](https://anaconda.org/conda-forge/turkish-lm-tuner)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/turkish-lm-tuner)](https://pypi.org/project/turkish-lm-tuner/)
[![Code license](https://img.shields.io/badge/Code%20License-MIT-green.svg)](https://github.com/boun-tabi-LMG/blob/main/LICENSE)
[![GitHub Repo stars](https://img.shields.io/github/stars/boun-tabi-lmt/safe)](https://github.com/boun-tabi-LMG/turkish-lm-tuner/stargazers)
[![arXiv](https://img.shields.io/badge/arxiv-custom-ID.svg)](https://arxiv.org/pdf/<arxiv-ID>.pdf)

## Overview 

Turkish LM Tuner is a library for fine-tuning Turkish language models on various NLP tasks. It is built on top of [HuggingFace Transformers](https://github.com/huggingface/transformers) library. It supports finetuning with conditional generation and sequence classification tasks. The library is designed to be modular and extensible. It is easy to add new tasks and models. The library also provides data loaders for various Turkish NLP datasets. 

## Installation

You can install the library from PyPI.

```bash
pip install turkish-lm-tuner
```

## Model Support

Any Encoder or ConditionalGeneration model that is compatible with HuggingFace Transformers library can be used with Turkish LM Tuner. The following models are tested and supported.
- [UL2TR](link to paper)
- [mT5](link to paper)
- [mBART](link to paper)
- [BERTurk](link to paper)

## Task and Dataset Support

| Task                           | Datasets                                                                                                 |
| ------------------------------ | -------------------------------------------------------------------------------------------------------- |
| Text Classification            | -                                                                                                        |
| Natural Language Inference     | [NLI_TR](reference), [SNLI_TR](reference), [MultiNLI_TR](reference)                                      |
| Semantic Textual Similarity    | [STSb_TR](reference)                                                                                     |
| Named Entity Recognition       | [WikiANN](reference), [Milliyet NER](reference)                                                          |
| Part-of-Speech Tagging         | [BOUN](reference), [IMST](reference)                                                                     |
| Question Answering             | [EXAMS](reference), [TQuAD](reference), [MKQA](reference)                                                |
| Text Summarization             | [TR News](reference), [MLSUM](reference), [Combined TR News and MLSUM](reference)                        |
| Title Generation               | [TR News](reference), [MLSUM](reference), [Combined TR News and MLSUM](reference)                        |
| Paraphrase Generation          | [OpenSubtitles](reference), [Tatoeba](reference), [TED Talks](reference)                                 |


## Usage
The tutorials in the [documentation]() can help you get started with `turkish-lm-tuner`.

## Examples

### Fine-tune and evaluate a conditional generation model 

```python
from turkish_lm_tuner import DatasetProcessor, TrainerForConditionalGeneration 

dataset_name = "tr_news" 
task = "summarization"
task_format="conditional_generation"
model_name = "boun-tabi-lmt/ul2tr"
max_input_length = 764
max_target_length = 128
dataset_processor = DatasetProcessor(
    dataset_name=dataset_name, task=task, task_format=task_format,
    model_name=model_name, max_input_length=max_input_length, max_target_length=max_target_length
)

train_dataset = dataset_processor.load_and_preprocess_data(split='train')
eval_dataset = dataset_processor.load_and_preprocess_data(split='validation')
test_dataset = dataset_processor.load_and_preprocess_data(split="test")

training_params = {
    'num_train_epochs': 10
    'per_device_train_batch_size': 4,
    'per_device_eval_batch_size': 4 
}

model_trainer = TrainerForConditionalGeneration(
    model_name=model_name, task=task,
    adafactor_scheduler=False, training_params=training_params, 
    model_save_path="ul2tr_tr_news_summarization", 
    dataset_name=dataset_name, 
    max_target_length=max_target_length, 
    postprocess_fn=dataset_processor.dataset.postprocess_data)

trainer, model = model_trainer.train_and_evaluate(train_dataset, eval_dataset, test_dataset)

model.save_pretrained(model_save_path)
dataset_processor.tokenizer.save_pretrained(model_save_path)
```

### Evaluate a conditional generation model with custom generation config

```python
from turkish_lm_tuner import DatasetProcessor, EvaluatorForConditionalGeneration

dataset_name = "tr_news" 
task = "summarization"
task_format="conditional_generation"
model_name = "boun-tabi-lmt/ul2tr"
max_input_length = 764
max_target_length = 128
dataset_processor = DatasetProcessor(
    dataset_name, task, task_format,
    model_name, max_input_length, max_target_length
)

test_dataset = dataset_processor.load_and_preprocess_data(split="test")

test_params = {
    'per_device_eval_batch_size': 4 
}

model_path = "ul2tr_tr_news_summarization"
generation_params = {
    'num_beams': 4,
    'length_penalty': 2.0,
    'no_repeat_ngram_size': 3,
    'early_stopping': True,
    'max_length': 128,
    'min_length': 30,
}
evaluator = EvaluatorForConditionalGeneration(
    model_path, task, max_target_length, test_params, 
    generation_params, dataset_processor.dataset.postprocess_data
)
results = evaluator.evaluate_model(test_dataset)
print(results)
```

## Reference

If you use this repository, please cite the following related [paper]():
```bibtex
@article{,
  title={UL2TR: A Large-Scale Pretrained Language Model for Turkish},
  author={},
  journal={},
  year={},
  publisher={}
}
```


## License

Note that all datasets belong to their respective owners. If you use the datasets provided by this library, please cite the original source.

This code base is licensed under the MIT license. See [LICENSE](license.md) for details.
