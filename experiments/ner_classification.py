import numpy as np
import evaluate
from datasets import load_dataset
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer, EarlyStoppingCallback

seqeval = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [str(p) for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [str(l) for (_, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


from turkish_lm_tuner.tr_datasets import WikiANNDataset

tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
dataset = WikiANNDataset()
train_dataset = dataset.load_dataset('train')
val_dataset = dataset.load_dataset('validation')
test_dataset = dataset.load_dataset('test')

processed_dataset = train_dataset.map(dataset.preprocess_data, batched=True, fn_kwargs={"skip_output_processing": True, "tokenizer": tokenizer})
processed_dataset_val = val_dataset.map(dataset.preprocess_data, batched=True, fn_kwargs={"skip_output_processing": True, "tokenizer": tokenizer})
processed_dataset_test = test_dataset.map(dataset.preprocess_data, batched=True, fn_kwargs={"skip_output_processing": True, "tokenizer": tokenizer})

model = AutoModelForTokenClassification.from_pretrained(
    "dbmdz/bert-base-turkish-cased", num_labels=7
)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir='/stratch/bounllm/finetuned-models/berturk/ner/wikiann/',
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

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
    eval_dataset=processed_dataset_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]

)

trainer.train()
trainer.evaluate(processed_dataset_test)