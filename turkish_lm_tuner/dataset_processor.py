from transformers import AutoTokenizer
import datasets
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
formatter = logging.Formatter('%(levelname)s - %(asctime)s - %(name)s: %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

from .tr_datasets import initialize_dataset

class DatasetProcessor:
    """
    Class for loading and preprocessing datasets

    Args:
        dataset_name: Name of the dataset to be loaded. See tr_datasets.py for list of available datasets
        task: Task to be performed on the dataset.
        task_format: Format of the task. Either 'classification' or 'conditional_generation'
        task_mode: Mode of the task. Either '', '[NLU]', '[NLG]' or '[S2S]'
        tokenizer_name: Name of the tokenizer to be used
        max_input_length: Maximum length of the input sequence
        max_target_length: Maximum length of the target sequence
        dataset_loc: Location of the dataset if it is not available in the HuggingFace Hub. A List must be given for multiple datasets, having None for HuggingFace datasets in the correspoding position.
        use_textual_output: Control for the utilization of textual output for Text-to-Text models.
    """
    def __init__(self,
                 dataset_name: str = None,
                 task: str = None,
                 task_format: str = None,
                 task_mode: str = '',
                 tokenizer_name: str = None,
                 max_input_length: int = None,
                 max_target_length: int = None,
                 dataset_loc: str = '',
                 use_textual_output: bool = False):

        logger.info(f"Initializing dataset processor for {dataset_name} dataset with {tokenizer_name} tokenizer and {task} task in {task_format} format with {task_mode} mode")
        logger.info(f"Max input length: {max_input_length} Max target length: {max_target_length} Utilizing Textual Output: {use_textual_output}")
        logger.info(f"Utilizing Textual Output: {use_textual_output}")

        self.dataset_name = dataset_name
        self.task = task
        self.task_format = task_format
        self.task_mode = task_mode
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.dataset_loc = dataset_loc
        self.use_textual_output = use_textual_output

        # Forcing the task format to be generation for Text-to-Text models
        if self.use_textual_output:
            self.task_format = "conditional_generation"


    def load_and_preprocess_data(self, split="train"):
        """
        Wrapper function for loading and preprocessing datasets (supporting multiple datasets, if dataset names are given in a list)
        Args:
            split: Split of the dataset to be loaded. Either 'train', 'validation' or 'test'
        """
        if self.use_textual_output and type(self.dataset_name) == list:
            tokenized_datasets = []
            for didx, dname in enumerate(self.dataset_name):
                if self.dataset_loc != "" and self.dataset_loc[didx] is not None:
                    tokenized_datasets.append(self.load_and_preprocess_singular_data(dname, self.dataset_loc[didx], split))
                else:
                    tokenized_datasets.append(self.load_and_preprocess_singular_data(dname, split=split))

            return datasets.concatenate_datasets(tokenized_datasets)
        else:

            return self.load_and_preprocess_singular_data(self.dataset_name, self.dataset_loc, split)


    def load_and_preprocess_singular_data(self, dataset_name, dataset_loc=None, split="train"):
        """
        Loads and preprocesses the dataset
        Args:
            dataset_name: Name of the dataset to be loaded. See tr_datasets.py for list of available datasets
            dataset_loc: Location of the dataset if it is not available in the HuggingFace Hub
            split: Split of the dataset to be loaded. Either 'train', 'validation' or 'test'
        """
        logger.info(f"Loading {split} split of {dataset_name} dataset")
        dataset = initialize_dataset(dataset_name, dataset_loc, self.use_textual_output)
        data = dataset.load_dataset(split)

        # setting dataset attribute, compliance for previous versions
        if not self.use_textual_output:
            self.dataset = dataset

        logger.info(f"Preprocessing {dataset_name} dataset")
        preprocess_function = dataset.preprocess_data

        column_names = data.column_names
        column_names = [col for col in column_names if col not in ['input_text', 'target_text', 'label', 'input_ids', 'label_ids']]

        if self.task_format == "classification":
            if self.task in ["ner", "pos_tagging"]:
                # Tokenize inputs and labels simultaneously
                processed_dataset = data.map(preprocess_function, remove_columns=column_names, batched=True, fn_kwargs={"skip_output_processing": True, "tokenizer": self.tokenizer})
                if "token_type_ids" in processed_dataset.column_names:
                    processed_dataset = processed_dataset.remove_columns("token_type_ids")
                return processed_dataset
            else:
                processed_dataset = data.map(preprocess_function, remove_columns=column_names, batched=True, fn_kwargs={"skip_output_processing": True})
        else:
            processed_dataset = data.map(preprocess_function, remove_columns=column_names, batched=True)

        if self.max_input_length == -1 or self.max_target_length == -1:
            # Compute token length statistics
            self.compute_token_length(processed_dataset)
            return

        logger.info(f"Tokenizing {dataset_name} dataset")
        tokenized_dataset = processed_dataset.map(self.tokenize_function, batched=True)
        return tokenized_dataset

    def compute_token_length(self, dataset):
        """
        Computes token length statistics for the dataset
        Args:
            dataset: Dataset to be processed.
        """

        def get_max_length(examples):
            return {
                'input_len': [len(ex) for ex in self.tokenizer(examples['input_text'])['input_ids']],
                'target_len': [len(ex) for ex in self.tokenizer(examples['target_text'])['input_ids']]
            }

        dataset = dataset.map(get_max_length, batched=True, batch_size=8)

        input_lengths = [length['input_len'] for length in dataset]
        target_lengths = [length['target_len'] for length in dataset]

        stats = {
                "Mean": np.mean,
                "Max": max,
                "90th percentile": lambda x: np.percentile(x, 90),
                "95th percentile": lambda x: np.percentile(x, 95),
                "99th percentile": lambda x: np.percentile(x, 99),
                "99.9th percentile": lambda x: np.percentile(x, 99.9)
            }

        for stat_name, func in stats.items():
            input_stat = func(input_lengths)
            target_stat = func(target_lengths)
            logger.info(f"{stat_name} input length: {input_stat}")
            logger.info(f"{stat_name} target length: {target_stat}")

    def prepend_prefix(self, examples):
        """
        Prepends task mode to the input text
        Args:
            examples: List of input texts
        """
        return [f'{self.task_mode}{ex}' for ex in examples]

    def append_eos(self, examples):
        """
        Appends EOS token to the input text
        Args:
            examples: List of input texts
        """
        def append_eos_text(text):
            if text.endswith(self.tokenizer.eos_token):
                return text
            else:
                return f'{text}{self.tokenizer.eos_token}'

        return [append_eos_text(ex) for ex in examples]

    def tokenize_function(self, examples, return_tensors=None):
        """
        Tokenizes the input and target texts
        Args:
            examples: List of input
        """
        if "input_ids" in examples:
            #examples["input_ids"] = [inputs + [self.tokenizer.pad_token_id] * (self.max_input_length - len(inputs)) if len(inputs) < self.max_input_length else inputs[:self.max_input_length] for inputs in examples["input_ids"]]
            #examples["label_ids"] = [label + [-100] * (self.max_input_length - len(label)) if len(label) < self.max_input_length else label[:self.max_input_length] for label in examples["label_ids"]]
            return examples

        if self.task_format == 'conditional_generation':
            inputs_tokenized = self.tokenizer(
                        self.prepend_prefix(examples["input_text"]),
                        padding="max_length",
                        truncation=True,
                        max_length=self.max_input_length,
                        return_token_type_ids=False,
                        return_tensors=return_tensors,
                   )

            if "target_text" not in examples:
                return inputs_tokenized

            targets_tokenized = self.tokenizer(
                        self.append_eos(examples["target_text"]),
                        padding="max_length",
                        truncation=True,
                        max_length=self.max_target_length,
                        return_token_type_ids=False,
                        return_tensors=return_tensors,
                   )
            return {'labels': targets_tokenized['input_ids'], **inputs_tokenized}
        elif self.task_format == 'classification':
            if self.tokenizer.eos_token == None:
                logger.info("No EOS token, don't append EOS token")
                return self.tokenizer(
                    examples["input_text"],
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_input_length,
                    return_token_type_ids=False,
                    return_tensors=return_tensors,
                )
            else:
                logger.info("EOS token present, append EOS token")
                return self.tokenizer(
                    self.append_eos(self.prepend_prefix(examples["input_text"])),
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_input_length,
                    return_token_type_ids=False,
                    return_tensors=return_tensors,
                )
