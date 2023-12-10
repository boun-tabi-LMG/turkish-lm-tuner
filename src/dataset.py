import datasets
from transformers import AutoTokenizer
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
formatter = logging.Formatter('%(levelname)s - %(asctime)s - %(name)s: %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

import os, json, re
from utils import (
    default_preprocess_function,
    preprocess_trnews_summarization,
    preprocess_trnews_title_generation,
    preprocess_mlsum_summarization,
    preprocess_mlsum_title_generation,
    preprocess_paraphrasing,
    preprocess_nli,
    preprocess_exams_qa, 
    preprocess_exams_qg,
    preprocess_mkqa_qa, 
    preprocess_mkqa_qg,
    preprocess_wikiann_ner,
    preprocess_ner_milliyet,
    preprocess_sts,
    postprocess_text
)

dataset_mapping = {
    "offensive": "Toygar/turkish-offensive-language-detection",

    # summarization/title generation
    "tr_news": "batubayk/TR-News",
    "mlsum": ("mlsum", "tu"),
    "combined_news": ["tr_news", "mlsum"], 

    # paraphrasing
    "opensubtitles": "mrbesher/tr-paraphrase-opensubtitles2018",
    "tatoeba": "mrbesher/tr-paraphrase-tatoeba",
    "ted": "mrbesher/tr-paraphrase-ted2013",

    # question answering & generation
    "exams": ("exams", "crosslingual_tr"),
    "mkqa": "mkqa",
    "turkish-nlp-qa-dataset": "furkanakkurt1618/qa_dataset-turkish-nlp-qa-dataset-boun-llm",
    "turkish-nlp-qa-dataset-qg": "furkanakkurt1618/qg_dataset-turkish-nlp-qa-dataset-boun-llm", # wasn't on hf

    # nli
<<<<<<< HEAD
<<<<<<< HEAD
    "snli_tr": ("nli_tr", "snli_tr"),
    "multinli_tr": ("nli_tr", "multinli_tr"),
    "nli_tr": ["snli_tr", "multinli_tr"], # SNLI and Multi-NLI merged together
=======
    "nli_tr": ["snli_tr", "multinli_tr"],
>>>>>>> 9357506 (Added nli conf and preprocessing)
=======
    "snli_tr": ("nli_tr", "snli_tr"),
    "multinli_tr": ("nli_tr", "multinli_tr"),
    "nli_tr": ["snli_tr", "multinli_tr"], # SNLI and Multi-NLI merged together
>>>>>>> 92a002e (Added Snli and multinli preprocessing)

    # semantic textual similarity
    "stsb_tr": {'train': 'stsb_tr_train.tsv', 'test': 'stsb_tr_test.tsv', 'validation': 'stsb_tr_dev.tsv'},

    # ner
    "milliyet": {'train': 'train.txt', 'test': 'test.txt', 'validation': 'dev.txt'},
    "wikiann": ("wikiann", "tr"),

    # pos tagging
    "boun": {'train': 'tr_boun-ud-train.conllu', 'test': 'tr_boun-ud-test.conllu', 'validation': 'tr_boun-ud-dev.conllu'},
    "imst": {'train': 'tr_imst-ud-train.conllu', 'test': 'tr_imst-ud-test.conllu', 'validation': 'tr_imst-ud-dev.conllu'},

    # text classification

}



class DatasetProcessor:
    def __init__(self, dataset_name, task, task_format, task_mode, tokenizer_name, max_input_length, max_target_length, dataset_loc="", no_preprocess=False):
        logger.info(f"Initializing dataset processor for {dataset_name} dataset with {tokenizer_name} tokenizer and {task} task in {task_format} format with {task_mode} mode")
        logger.info(f"Max input length: {max_input_length} Max target length: {max_target_length}")
        self.dataset_name = dataset_name
        self.task = task
        self.task_format = task_format
        self.task_mode = task_mode
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.dataset_loc = dataset_loc
        self.no_preprocess = no_preprocess

    def load_and_preprocess_data(self, split='train'):
        logger.info(f"Loading {split} split of {self.dataset_name} dataset")
        mapped_dataset = dataset_mapping[self.dataset_name]
        # For HF datasets with two dataset specifications (i.e. ("wikiann", "tr"))
        if type(mapped_dataset) == tuple:
            if "multinli" in self.dataset_name and split == "test":
                split = "validation_matched" # There's no test set in Multi-NLI
            dataset = datasets.load_dataset(mapped_dataset[0], mapped_dataset[1], split=split)
            if "nli" in self.dataset_name:
                dataset = dataset.filter(lambda example: example["label"] != -1) # removed samples with the label -1 

        # For local datasets (need to specify dataset location in .yaml file)
        elif self.dataset_name == "milliyet":
            data_files = [i for i in os.listdir(self.dataset_loc) if i.endswith('.txt') and i.startswith(split)]
            data_file = data_files[0]
            dataset_dict = {}
            dataset_dict[split] = []
            with open(os.path.join(self.dataset_loc, data_file), 'r', encoding='utf-8') as f:
                content = f.read()
            data = content.split('\n\n')
            for example in data:
                if example.strip() == '':
                    continue
                lines = example.split('\n')
                tokens = []
                tags = []
                for line in lines:
                    if line.strip() == '':
                        break
                    token, tag = line.split(' ')
                    tokens.append(token)
                    tags.append(tag)
                el = {'tokens': tokens, 'tags': tags}
                dataset_dict[split].append(el)
            with open(os.path.join(self.dataset_loc, split + '.json'), 'w', encoding='utf-8') as f:
                json.dump(dataset_dict[split], f, ensure_ascii=False)
            mapped_dataset = {split: split + '.json'}
            dataset = datasets.load_dataset(self.dataset_loc, data_files=mapped_dataset, split=split)
        elif self.dataset_name in ["boun", "imst"]:
            md_pattern = re.compile('^# (.+?) = (.+?)$')
            annotation_pattern = re.compile('(.+\t){9}.+')
            data_d = {'treebank_name': self.dataset_name, 'sentences': {}}
            data_file = os.path.join(self.dataset_loc, mapped_dataset[split])
            with open(data_file, 'r', encoding='utf-8') as f:
                content = f.read()
            sents = content.split('\n\n')
            for sent in sents:
                lines = sent.split('\n')
                sent_id = ''
                d_t = {}
                for i, line in enumerate(lines):
                    md_match = md_pattern.match(line)
                    if md_match:
                        field = md_match.group(1).strip()
                        value = md_match.group(2).strip()
                        if field == 'sent_id':
                            sent_id = value
                        else:
                            d_t[field] = value
                    annotation_match = annotation_pattern.match(line)
                    if annotation_match:
                        annotation = '\n'.join(lines[i:])
                        d_t['table'] = annotation
                        d_t['split'] = split
                        break
                if d_t:
                    data_d['sentences'][sent_id] = d_t
            pos_d_tr = { "ADP": "edat", "AUX": "yardımcı", "PRON": "zamir", "NOUN": "isim", "PROPN": "özel", "INTJ": "ünlem", "PART": "tanımcık", "CCONJ": "eşgüdümlü", "VERB": "fiil", "SYM": "sembol", "DET": "belirteç", "ADV": "zarf", "ADJ": "sıfat", "X": "diğer", "SCONJ": "yantümce", "NUM": "sayı", "PUNCT": "noktalama" }
            new_l = []
            sentences = data_d['sentences']
            for sent_id in sentences:
                table = sentences[sent_id]['table']
                text = sentences[sent_id]['text']
                split_t = sentences[sent_id]['split']
                tag_l = []
                split_token = 0
                for row in table.split('\n'):
                    if row:
                        fields = row.split('\t')
                        id_t, form, pos = fields[0], fields[1], fields[3]
                        if '-' in id_t:
                            split_token = 2
                        if pos == '_':
                            continue
                        if split_token == 1:
                            tag_l.append('-{}/{}'.format(form, pos_d_tr[pos]))
                        else:
                            tag_l.append('{}/{}'.format(form, pos_d_tr[pos]))
                        if split_token != 0:
                            split_token -= 1
                output = ' '.join(tag_l)
                new_l.append({'target_text': output, 'sent_id': sent_id, 'input_text': text})
            with open(os.path.join(self.dataset_loc, split + '.json'), 'w', encoding='utf-8') as f:
                json.dump(new_l, f, ensure_ascii=False)
            dataset = datasets.load_dataset(self.dataset_loc, data_files={split: split + '.json'}, split=split)
        elif type(mapped_dataset) == dict:
            dataset = datasets.load_dataset(self.dataset_loc, data_files=mapped_dataset, split=split)
        # For the NLI_TR HF dataset
        elif self.dataset_name == "nli_tr":
            if split == "train":
                mnli_dataset = datasets.load_dataset("nli_tr", 'multinli_tr', split="train")
                snli_dataset = datasets.load_dataset("nli_tr", 'snli_tr', split="train")
                snli_dataset = snli_dataset.filter(lambda example: example["label"] != -1) # removed samples with the label -1 (785 samples in train)
                dataset = datasets.concatenate_datasets([mnli_dataset, snli_dataset])
            else:
                dataset = datasets.load_dataset("nli_tr", 'snli_tr', split=split)
                dataset = dataset.filter(lambda example: example["label"] != -1) # removed samples with the label -1 
        elif self.dataset_name == 'combined_news':
            tr_news_dataset = datasets.load_dataset(dataset_mapping["tr_news"], split=split)
            mlsum_dataset = datasets.load_dataset("mlsum", 'tu', split=split)
            mlsum_dataset = mlsum_dataset.rename_column("text", "content")
            mlsum_dataset = mlsum_dataset.rename_column("summary", "abstract")
            dataset = datasets.concatenate_datasets([tr_news_dataset, mlsum_dataset])

        # For HF datasets with a single dataset specification (i.e. "nli_tr")
        else:
            dataset = datasets.load_dataset(mapped_dataset, split=split) #.select(range(100))
        if self.no_preprocess:
            processed_dataset = dataset
        else:
            logger.info(f"Preprocessing {self.dataset_name} dataset")
            preprocess_function = self.get_preprocess_function()
            column_names = dataset.column_names
            column_names = [col for col in column_names if col not in ['input_text', 'target_text', 'label']]
            if self.task_format == "classification" and self.task == "nli":
                processed_dataset = dataset.map(preprocess_function, remove_columns=column_names, batched=True, fn_kwargs={"skip_output_processing": True})
            else:
                processed_dataset = dataset.map(preprocess_function, remove_columns=column_names, batched=True)
        if self.max_input_length == -1 or self.max_target_length == -1:
            self.compute_token_length(processed_dataset)
            return
        
        logger.info(f"Tokenizing {self.dataset_name} dataset")
        tokenized_dataset = processed_dataset.map(self.tokenize_function, batched=True)
        return tokenized_dataset

    def compute_token_length(self, dataset):

        def get_max_length(examples):
            return {
                'input_len': [len(ex) for ex in self.tokenizer(examples['input_text'])['input_ids']],
                'target_len': [len(ex) for ex in self.tokenizer(examples['target_text'])['input_ids']]
            }

        dataset = dataset.map(get_max_length, batched=True, batch_size=8)
        max_input_length = max([i['input_len'] for i in dataset])
        max_target_length = max([i['target_len'] for i in dataset])
        mean_input_length = sum([i['input_len'] for i in dataset])/len(dataset)
        mean_target_length = sum([i['target_len'] for i in dataset])/len(dataset)
        print(f"Mean input length: {mean_input_length} Mean target length: {mean_target_length}")
        print(f"Max input length: {max_input_length} Max target length: {max_target_length}")    
        input_percent90 = np.percentile([i['input_len'] for i in dataset], 90)
        target_percent90 = np.percentile([i['target_len'] for i in dataset], 90)
        print(f"90th percentile input length: {input_percent90} 90th percentile target length: {target_percent90}")
        input_percent95 = np.percentile([i['input_len'] for i in dataset], 95)
        target_percent95 = np.percentile([i['target_len'] for i in dataset], 95)
        print(f"95th percentile input length: {input_percent95} 95th percentile target length: {target_percent95}")
        input_percent99 = np.percentile([i['input_len'] for i in dataset], 99)
        target_percent99 = np.percentile([i['target_len'] for i in dataset], 99)
        print(f"99th percentile input length: {input_percent99} 99th percentile target length: {target_percent99}")
        input_percent999 = np.percentile([i['input_len'] for i in dataset], 99.9)
        target_percent999 = np.percentile([i['target_len'] for i in dataset], 99.9)
        print(f"99.9th percentile input length: {input_percent999} 99.9th percentile target length: {target_percent999}")
    
    def get_preprocess_function(self):
        # Mapping of dataset_name and task to corresponding preprocess functions
        preprocess_functions = {
            ('tr_news', 'summarization'): preprocess_trnews_summarization,
            ('tr_news', 'title_generation'): preprocess_trnews_title_generation,
            ('mlsum', 'summarization'): preprocess_mlsum_summarization,
            ('mlsum', 'title_generation'): preprocess_mlsum_title_generation,
            ('combined_news', 'summarization'): preprocess_trnews_summarization,
            ('combined_news', 'title_generation'): preprocess_trnews_title_generation,
            ('opensubtitles', 'paraphrasing'): preprocess_paraphrasing,
            ('ted', 'paraphrasing'): preprocess_paraphrasing,
            ('tatoeba', 'paraphrasing'): preprocess_paraphrasing,
            ('exams', 'question_answering'): preprocess_exams_qa,
            ('exams', 'question_generation'): preprocess_exams_qg,
            ("mkqa", "question_answering"): preprocess_mkqa_qa,
            ("mkqa", "question_generation"): preprocess_mkqa_qg,
            ("wikiann", "ner"): preprocess_wikiann_ner,
            ("milliyet", "ner"): preprocess_ner_milliyet,
            ("stsb_tr", "semantic_similarity") : preprocess_sts,
            ("nli_tr", "nli") : preprocess_nli,
            ("snli_tr", "nli") : preprocess_nli,
            ("multinli_tr", "nli") : preprocess_nli,
            # ... add mappings for other dataset and task type combinations
        }
        return preprocess_functions.get((self.dataset_name, self.task), default_preprocess_function)
    
    def prepend_prefix(self, examples):
        return [f'{self.task_mode}{ex}' for ex in examples]
    
    def append_eos(self, examples):
        def append_eos_text(text):
            if text.endswith(self.tokenizer.eos_token):
                return text
            else:
                return f'{text}{self.tokenizer.eos_token}'

        return [append_eos_text(ex) for ex in examples]

    def tokenize_function(self, examples):
        if self.task_format == 'conditional_generation':
            inputs_tokenized = self.tokenizer(
                        self.prepend_prefix(examples["input_text"]),
                        padding="max_length",
                        truncation=True,
                        max_length=self.max_input_length,
                        return_token_type_ids=False,
                   )
            targets_tokenized = self.tokenizer(
                        self.append_eos(examples["target_text"]),
                        padding="max_length",
                        truncation=True,
                        max_length=self.max_target_length,
                        return_token_type_ids=False,
                   )
            return {'labels': targets_tokenized['input_ids'], **inputs_tokenized}
        return self.tokenizer(
            self.append_eos(self.prepend_prefix(examples["input_text"])),
            padding="max_length",
            truncation=True,
            max_length=self.max_input_length,
            return_token_type_ids=False,
        )
