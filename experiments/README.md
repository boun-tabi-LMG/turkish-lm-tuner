# Experiments

This directory contains the scripts for fine-tuning and evaluating models on various Turkish NLP tasks. The experiments are run through `finetune.py` and `eval.py` scripts. The experiments are configured through YAML files in the `conf/` and `generation_conf` directories. These files contain the hyperparameters and other configuration parameters for the experiments. Each task and optionally dataset has a separate configuration file. The configuration files are named as `<task>.yaml` and `<task>_<dataset>.yaml` respectively.


## Installation

The experiments require installing `turkish-lm-tuner` library. The library can be installed from PyPI.

```bash
pip install turkish-lm-tuner
```

## Fine-tuning

To fine-tune a model on a specific task, run the following command by replacing `<task>` with the name of the task, `<task>_<dataset>` if dataset has specific parameters.

```bash
python experiments/finetune.py --config-name <task>
```

By default, the script will use the parameters in the configuration file, where the default model is `TURNA`. You can override the parameters by passing them as command line arguments. For example, to override the `model_name` parameter, run the following command.

```bash
python experiments/finetune.py --config-name <task> model_name=google/mt5-large training_params.output_dir=<output_dir>
```

## Evaluation

To evaluate a fine-tuned model on a specific task, run the following command by replacing `<task>` with the name of the task, `<task>_<dataset>` if dataset has specific parameters.

```bash
python experiments/eval.py --config-name <task>
```



