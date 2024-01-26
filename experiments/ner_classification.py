import numpy as np
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer, EarlyStoppingCallback
from turkish_lm_tuner.tr_datasets import WikiANNDataset
from turkish_lm_tuner.metrics import load_task_metrics
from turkish_lm_tuner.metrics import Evaluator

tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
dataset = WikiANNDataset()
wikiann = dataset.load_dataset()
processed_dataset = wikiann.map(dataset.preprocess_data, batched=True, fn_kwargs={"skip_output_processing": True, "tokenizer": tokenizer})

model = AutoModelForTokenClassification.from_pretrained(
    "dbmdz/bert-base-turkish-cased", num_labels=7
)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir='berturk/ner/wikiann/',
    metric_for_best_model='eval_loss',
    greater_is_better=False,
    num_train_epochs=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_dir="./logs",
    logging_steps=100,
    save_total_limit=1,
    push_to_hub=False,
    report_to="wandb"
)

# Task metric can be loaded with `load_task_metrics` function
metric = load_task_metrics("ner")[0]
# Alternatively, `Evaluator` class can be used to load all task metrics 
# eval = Evaluator(task='ner')

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset["train"],
    eval_dataset=processed_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=lambda x: metric.compute(*x),
    #compute_metrics=lambda x: eval.compute_metrics(*x),
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]

)

trainer.train()
print(trainer.evaluate(processed_dataset["test"]))