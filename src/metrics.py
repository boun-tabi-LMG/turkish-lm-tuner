import evaluate
import sys 

class BaseMetric:
    def __init__(self, metric_name):
        self.metric_name = metric_name
        self.metric = self.load_metric()

    def load_metric(self):
        return evaluate.load(self.metric_name)
    
    def compute(self, preds, labels, **kwargs):
        return self.metric.compute(predictions=preds, references=labels, **kwargs)

class Accuracy(BaseMetric):
    def __init__(self):
        super().__init__("accuracy")

class Precision(BaseMetric):
    def __init__(self):
        super().__init__("precision")
    
class Recall(BaseMetric):
    def __init__(self):
        super().__init__("recall")

class F1(BaseMetric):
    def __init__(self):
        super().__init__("f1")

class Pearsonr(BaseMetric):
    def __init__(self):
        super().__init__("pearsonr")

class BLEU(BaseMetric):
    def __init__(self):
        super().__init__("bleu")

class METEOR(BaseMetric):
    def __init__(self):
        super().__init__("meteor")

class ROUGE(BaseMetric):
    def __init__(self):
        super().__init__("rouge")

class TER(BaseMetric):
    def __init__(self):
        super().__init__("ter")


METRIC_MAPPING_NAMES = [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
        ("pearsonr", "Pearsonr"),
        ("bleu", "BLEU"),
        ("meteor", "METEOR"),
        ("rouge", "ROUGE"),
        ("ter", "TER"),
    ]

def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)

def load_metrics(metrics):
    def load_metric(metric):
        for metric_mapping_name in METRIC_MAPPING_NAMES:
            if metric == metric_mapping_name[0]:
                metric_class = str_to_class(metric_mapping_name[1])
                return metric_class()
        raise NotImplementedError(f"Metric {metric} not implemented. Must be one of {METRIC_MAPPING_NAMES}")
    return [load_metric(metric.lower()) for metric in metrics]

def load_task_metrics(task):
    if task == "classification":
        return load_metrics(["accuracy", "precision", "recall", "f1"])
    elif task in ["summarization", "paraphrasing", "title_generation"]:
        return load_metrics(["rouge", "bleu", "meteor", "ter"])
    elif task == "nli":
        return load_metrics(["accuracy"])
    elif task == "semantic_similarity":
        return load_metrics(["pearsonr"])
    else:
        raise NotImplementedError(f"Task {task} not implemented.")