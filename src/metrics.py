import evaluate
import sys 

class BaseMetric:
    """
    A base class for different evaluation metrics.
    
    Attributes:
        metric_name (str): The name of the metric.
        metric (evaluate.Metric): An instance of the evaluate metric.
    """
    def __init__(self, metric_name):
        """
        Initializes the BaseMetric class with a given metric name.

        Args:
            metric_name (str): The name of the metric to load.
        """
        self.metric_name = metric_name
        self.metric = self.load_metric()

    def load_metric(self):
        """
        Loads the evaluation metric using the 'evaluate' library.

        Returns:
            evaluate.Metric: An instance of the evaluation metric.
        """
        return evaluate.load(self.metric_name)
    
    def compute(self, preds, labels, **kwargs):
        """
        Computes the metric score.

        Args:
            preds (list): A list of predictions.
            labels (list): A list of ground truth labels.
            **kwargs: Additional keyword arguments for the metric computation.

        Returns:
            dict: A dictionary containing the metric score.
        """
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

class F1Macro(BaseMetric):
    def __init__(self):
        super().__init__("f1_macro")

    def compute(self, preds, labels):
        return super().compute(predictions=preds, references=labels, average="macro")
    
class F1Micro(BaseMetric):
    def __init__(self):
        super().__init__("f1_micro")

    def compute(self, preds, labels):
        return super().compute(predictions=preds, references=labels, average="micro")
    
class F1Weighted(BaseMetric):
    def __init__(self):
        super().__init__("f1_weighted")

    def compute(self, preds, labels):
        return super().compute(predictions=preds, references=labels, average="weighted")
    
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
    """
    Converts a string to a class object.

    Args:
        classname (str): The name of the class.

    Returns:
        type: The class object corresponding to the classname.
    """
    return getattr(sys.modules[__name__], classname)

def load_metrics(metrics):
    """
    Loads a list of metric objects based on their names.

    Args:
        metrics (list): A list of metric names as strings.

    Returns:
        list: A list of metric class instances.
    """
    def load_metric(metric):
        for metric_mapping_name in METRIC_MAPPING_NAMES:
            if metric == metric_mapping_name[0]:
                metric_class = str_to_class(metric_mapping_name[1])
                return metric_class()
        raise NotImplementedError(f"Metric {metric} not implemented. Must be one of {METRIC_MAPPING_NAMES}")
    return [load_metric(metric.lower()) for metric in metrics]

def load_task_metrics(task):
    """
    Loads metrics relevant to a specific task.

    Args:
        task (str): The name of the task.

    Returns:
        list: A list of metric class instances relevant to the task.
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