<h1 align="center">  🦖 Turkish LM Tuner </h1>
<!--<h4 align="center"> Summary of project or library comes here. </h4>-->

</br>

[![PyPI](https://img.shields.io/pypi/v/turkish-lm-tuner)](https://pypi.org/project/turkish-lm-tuner/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/turkish-lm-tuner)](https://pypi.org/project/turkish-lm-tuner/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/turkish-lm-tuner)](https://pypi.org/project/turkish-lm-tuner/)
[![Code license](https://img.shields.io/badge/Code%20License-MIT-green.svg)](https://github.com/boun-tabi-LMG/blob/main/LICENSE)
[![GitHub Repo stars](https://img.shields.io/github/stars/boun-tabi-LMG/turkish-lm-tuner)](https://github.com/boun-tabi-LMG/turkish-lm-tuner/stargazers)
[![arXiv](https://img.shields.io/badge/arxiv-2401.14373-b31b1b.svg)](https://arxiv.org/abs/2401.14373)

## Overview

Turkish LM Tuner is a library for fine-tuning Turkish language models on various NLP tasks. It is built on top of [HuggingFace Transformers](https://github.com/huggingface/transformers) library. It supports finetuning with conditional generation and sequence classification tasks. The library is designed to be modular and extensible. It is easy to add new tasks and models. The library also provides data loaders for various Turkish NLP datasets.

## Installation

You can use the following command to install the library:

```bash

pip install git+https://github.com/boun-tabi-LMG/turkish-lm-tuner.git
```

## Model Support

Any Encoder or ConditionalGeneration model that is compatible with HuggingFace Transformers library can be used with Turkish LM Tuner. The following models are tested and supported.

- [TURNA](https://arxiv.org/abs/2401.14373)
- [mT5](https://aclanthology.org/2021.naacl-main.41/)
- [mBART](https://aclanthology.org/2020.tacl-1.47/)
- [BERTurk](https://github.com/stefan-it/turkish-bert)

## Task and Dataset Support

| Task                           | Datasets                                                                                                 |
| ------------------------------ | --------------------------------------------------------------------------------------------------------                                                                                                             |
| Text Classification            | [Product Reviews](https://huggingface.co/datasets/turkish_product_reviews), [TTC4900](https://dx.doi.org/10.5505/pajes.2018.15931), [Tweet Sentiment](https://ieeexplore.ieee.org/document/8554037)                  |                                                                                                                                 |
| Natural Language Inference     | [NLI_TR](https://aclanthology.org/2020.emnlp-main.662/), [SNLI_TR](https://aclanthology.org/2020.emnlp-main.662/), [MultiNLI_TR](https://aclanthology.org/2020.emnlp-main.662/)                                      |
| Semantic Textual Similarity    | [STSb_TR](https://aclanthology.org/2021.gem-1.3/)                                                                                     |
| Named Entity Recognition       | [WikiANN](https://aclanthology.org/P19-1015/), [Milliyet NER](https://doi.org/10.1017/S135132490200284X)                                                          |
| Part-of-Speech Tagging         | [BOUN](https://universaldependencies.org/treebanks/tr_boun/index.html), [IMST](https://universaldependencies.org/treebanks/tr_imst/index.html)                                                                     |
| Text Summarization             | [TR News](https://doi.org/10.1007/s10579-021-09568-y), [MLSUM](https://aclanthology.org/2020.emnlp-main.647/), [Combined TR News and MLSUM](https://doi.org/10.1017/S1351324922000195)                        |
| Title Generation               | [TR News](https://doi.org/10.1007/s10579-021-09568-y), [MLSUM](https://aclanthology.org/2020.emnlp-main.647/), [Combined TR News and MLSUM](https://doi.org/10.1017/S1351324922000195)                        |
| Paraphrase Generation          | [OpenSubtitles](https://aclanthology.org/2022.icnlsp-1.14/), [Tatoeba](https://aclanthology.org/2022.icnlsp-1.14/), [TED Talks](https://aclanthology.org/2022.icnlsp-1.14/)                                 |


## Usage
The tutorials in the [documentation](docs/) can help you get started with `turkish-lm-tuner`.

## Examples

### Fine-tune and evaluate a conditional generation model

```python
from turkish_lm_tuner import DatasetProcessor, TrainerForConditionalGeneration

dataset_name = "tr_news"
task = "summarization"
task_format="conditional_generation"
model_name = "boun-tabi-LMG/TURNA"
max_input_length = 764
max_target_length = 128
dataset_processor = DatasetProcessor(
    dataset_name=dataset_name, task=task, task_format=task_format, task_mode='',
    tokenizer_name=model_name, max_input_length=max_input_length, max_target_length=max_target_length
)

train_dataset = dataset_processor.load_and_preprocess_data(split='train')
eval_dataset = dataset_processor.load_and_preprocess_data(split='validation')
test_dataset = dataset_processor.load_and_preprocess_data(split="test")

training_params = {
    'num_train_epochs': 10
    'per_device_train_batch_size': 4,
    'per_device_eval_batch_size': 4,
    'output_dir': './', 
    'evaluation_strategy': 'epoch',
    'save_strategy': 'epoch',
    'predict_with_generate': True    
}
optimizer_params = {
    'optimizer_type': 'adafactor',
    'scheduler': False,
}

model_trainer = TrainerForConditionalGeneration(
    model_name=model_name, task=task,
    optimizer_params=optimizer_params,
    training_params=training_params,
    model_save_path="turna_summarization_tr_news",
    max_input_length=max_input_length,
    max_target_length=max_target_length, 
    postprocess_fn=dataset_processor.dataset.postprocess_data
)

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
model_name = "boun-tabi-LMG/TURNA"
task_mode = ''
max_input_length = 764
max_target_length = 128
dataset_processor = DatasetProcessor(
    dataset_name, task, task_format, task_mode,
    model_name, max_input_length, max_target_length
)

test_dataset = dataset_processor.load_and_preprocess_data(split="test")

test_params = {
    'per_device_eval_batch_size': 4
}

model_path = "turna_tr_news_summarization"
generation_params = {
    'num_beams': 4,
    'length_penalty': 2.0,
    'no_repeat_ngram_size': 3,
    'early_stopping': True,
    'max_length': 128,
    'min_length': 30,
}
evaluator = EvaluatorForConditionalGeneration(
    model_path, model_name, task, max_input_length, max_target_length, test_params,
    generation_params, dataset_processor.dataset.postprocess_data
)
results = evaluator.evaluate_model(test_dataset)
print(results)
```

## Reference

If you use this repository, please cite the following related [paper](https://arxiv.org/abs/2401.14373):

```bibtex
@misc{uludoğan2024turna,
      title={TURNA: A Turkish Encoder-Decoder Language Model for Enhanced Understanding and Generation}, 
      author={Gökçe Uludoğan and Zeynep Yirmibeşoğlu Balal and Furkan Akkurt and Melikşah Türker and Onur Güngör and Susan Üsküdarlı},
      year={2024},
      eprint={2401.14373},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## License

Note that all datasets belong to their respective owners. If you use the datasets provided by this library, please cite the original source.

This code base is licensed under the MIT license. See [LICENSE](license.md) for details.
