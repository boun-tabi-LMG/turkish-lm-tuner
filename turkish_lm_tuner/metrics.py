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

class PrecisionWeighted(BaseMetric):
    def __init__(self):
        super().__init__("precision")

    def compute(self, preds, labels):
        return self.metric.compute(predictions=preds, references=labels, average="weighted")
    
class Recall(BaseMetric):
    def __init__(self):
        super().__init__("recall")

class RecallWeighted(BaseMetric):
    def __init__(self):
        super().__init__("recall")

    def compute(self, preds, labels):
        return self.metric.compute(predictions=preds, references=labels, average="weighted")

class F1(BaseMetric):
    def __init__(self):
        super().__init__("f1")

class F1Macro(BaseMetric):
    def __init__(self):
        super().__init__("f1")

    def compute(self, preds, labels):
        return self.metric.compute(predictions=preds, references=labels, average="macro")
    
class F1Micro(BaseMetric):
    def __init__(self):
        super().__init__("f1")

    def compute(self, preds, labels):
        return self.metric.compute(predictions=preds, references=labels, average="micro")
    
class F1Weighted(BaseMetric):
    def __init__(self):
        super().__init__("f1")

    def compute(self, preds, labels):
        return self.metric.compute(predictions=preds, references=labels, average="weighted")
    
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

class SQUAD(BaseMetric):
    def __init__(self):
        super().__init__("squad")

    def compute(self, preds, labels, **kwargs):
        for i in range(len(labels)):
            label_t = labels[i]
            labels[i] = {"answers": {"answer_start": [0], "text": [label_t]}, "id": str(i)}
        for i in range(len(preds)):
            pred_t = preds[i]
            preds[i] = {'prediction_text': pred_t.strip(), 'id': str(i)}
        return self.metric.compute(predictions=preds, references=labels, **kwargs)

class SeqEval(BaseMetric):
    def __init__(self):
        super().__init__("seqeval")

    def compute(self, preds, labels, **kwargs):
        # if labels.shape != preds.shape:
        #     preds = np.argmax(preds, axis=-1)

        true_predictions = [
            [str(f'B-{p}') if len(str(p)) == 1 else p for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(preds, labels)
        ]
        true_labels = [
            [str(f'B-{l}') if len(str(l)) == 1 else l for (_, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(preds, labels)
        ]
        return self.metric.compute(predictions=true_predictions, references=true_labels)

METRIC_MAPPING_NAMES = [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("precision_weighted", "PrecisionWeighted"),
        ("recall", "Recall"),
        ("recall_weighted", "RecallWeighted"),
        ("f1", "F1"),
        ("f1_macro", "F1Macro"),
        ("f1_micro", "F1Micro"),
        ("f1_weighted", "F1Weighted"),
        ("pearsonr", "Pearsonr"),
        ("bleu", "BLEU"),
        ("meteor", "METEOR"),
        ("rouge", "ROUGE"),
        ("ter", "TER"),
        ("squad", "SQUAD"),
        ("seqeval", "SeqEval")
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
    """
    if task == "classification":
        return load_metrics(["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"])
    elif task in ["summarization", "paraphrasing", "title_generation"]:
        return load_metrics(["rouge", "bleu", "meteor", "ter"])
    elif task == "nli":
        return load_metrics(["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"])
    elif task == "semantic_similarity":
        return load_metrics(["pearsonr"])
    elif task == "ner":
        return load_metrics(["seqeval"])
    elif task == "pos_tagging":
        return load_metrics(["seqeval"])
    elif task == "question_answering":
        return load_metrics(["squad"])
    elif task == "question_generation":
        return load_metrics(["rouge", "bleu", "meteor", "ter"])
    else:
        raise NotImplementedError(f"Task {task} not implemented.")
    

import numpy as np

class Evaluator:
    """
    A class for evaluating predictions using multiple metrics.

    Attributes:
        metrics (list): A list of metric instances.
    """
    def __init__(self, task=None, metrics=None):
        """
        Initializes the Evaluator class.

        Args:
            task (str, optional): The name of the task for which to load metrics. Defaults to None.
            metrics (list, optional): A list of metric names to load. Defaults to None.

        Raises:
            ValueError: If neither task nor metrics are specified.
        """
        if task is not None: 
            self.metrics = load_task_metrics(task)
        else:
            if metrics is None:
                raise ValueError("Either task or metrics must be specified.")
            self.metrics = load_metrics(metrics)
    
    def compute_metrics(self, preds, labels):
        """
        Computes the metrics for the given predictions and labels.

        Args:
            preds (list): A list of predictions.
            labels (list): A list of ground truth labels.

        Returns:
            dict: A dictionary of metric scores.
        """
        scores = {}
        for metric in self.metrics:
            metric_scores = metric.compute(preds, labels)
            scores.update(metric_scores)
        return scores
    
    def compute_bootstrapped_metrics(self, preds, labels, num_samples=1000):
        """
        Computes bootstrapped metrics for uncertainty estimation.

        Args:
            preds (list): A list of predictions.
            labels (list): A list of ground truth labels.
            num_samples (int): The number of bootstrap samples to generate.

        Returns:
            tuple: A tuple containing two dictionaries, one for average scores and one for standard deviations.
        """
        scores = {metric_name: [] for metric_name in self.metrics}
        for _ in range(num_samples):
            # Generating indices for bootstrap samples
            sample_indices = np.random.choice(len(preds), size=len(preds), replace=True)
            sampled_preds = [preds[i] for i in sample_indices]
            sampled_labels = [labels[i] for i in sample_indices]

            # Computing metrics for the sampled data
            for metric in self.metrics:
                metric_scores = metric.compute(sampled_preds, sampled_labels)
                for key, value in metric_scores.items():
                    scores[key].append(value)

        # Calculating average and standard deviation for the metrics
        average_scores = {key: np.mean(values) for key, values in scores.items()}
        std_scores = {key: np.std(values) for key, values in scores.items()}

        return average_scores, std_scores
