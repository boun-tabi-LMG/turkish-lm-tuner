import datasets
import sys
import json
import re

from pathlib import Path

class BaseDataset:
    DATASET_NAME = None
    DATASET_INFO = None
    def __init__(self, dataset_name=None, dataset_info=None):
        if dataset_name is not None:
            self.dataset_name = dataset_name
        else:
            self.dataset_name = self.DATASET_NAME
        if dataset_info is not None:
            self.dataset_info = dataset_info
        else:
            self.dataset_info = self.DATASET_INFO

    def load_dataset(self, split=None):
        if type(self.dataset_info) == tuple:
            return datasets.load_dataset(self.dataset_info[0], self.dataset_info[1], split=split)
        elif type(self.dataset_info) == str:
            return datasets.load_dataset(self.dataset_info, split=split)
        else:
            raise NotImplementedError

    def preprocess_data(self, examples):
        return {"input_text": examples["text"], "label": examples["label"]}

    def postprocess_data(self, examples):
        return [ex.strip() if isinstance(ex, str) else ex for ex in examples]

    def deduplicate_data(self, dataset, input_column):
        df = dataset.to_pandas()
        df = df.drop_duplicates(subset=[input_column])
        dedup_dataset = datasets.Dataset.from_pandas(df)
        drop_columns = [col for col in dedup_dataset.column_names if col not in dataset.column_names]
        dedup_dataset = dedup_dataset.map(remove_columns=drop_columns)
        return dedup_dataset


class TRNewsDataset(BaseDataset):
    DATASET_NAME = "tr_news"
    DATASET_INFO = "batubayk/TR-News"

    def preprocess_data(self, examples):
        return {"input_text": examples["content"], "target_text": examples["abstract"]}

class MLSumDataset(BaseDataset):
    DATASET_NAME = "mlsum"
    DATASET_INFO = ("mlsum", "tu")

    def preprocess_data(self, examples):
        return {"input_text": examples["text"], "target_text": examples["summary"]}

class TRNewsTitleDataset(BaseDataset):
    DATASET_NAME = "tr_news"
    DATASET_INFO = "batubayk/TR-News"

    def preprocess_data(self, examples):
        return {"input_text": examples["content"], "target_text": examples["title"]}

class MLSumTitleDataset(BaseDataset):
    DATASET_NAME = "mlsum"
    DATASET_INFO = ("mlsum", "tu")

    def preprocess_data(self, examples):
        return {"input_text": examples["text"], "target_text": examples["title"]}

class CombinedNewsDataset(TRNewsDataset):
    DATASET_NAME = "combined_news"
    DATASET_INFO = ["tr_news", "mlsum"]

    def load_dataset(self, split=None):
        trnews = TRNewsDataset().load_dataset(split)
        mlsum = MLSumDataset().load_dataset(split)
        mlsum = mlsum.rename_column("text", "content")
        mlsum = mlsum.rename_column("summary", "abstract")
        if split is not None:
            return datasets.concatenate_datasets([trnews, mlsum])
        else:
            combined_data = {}
            for key in trnews.keys():
                combined_data[key] = datasets.concatenate_datasets([trnews[key], mlsum[key]])
            # Returns DatasetDict object which is compatible with other datasets but takes a lot of time
            # return datasets.Dataset.from_dict(combined_data)
            # Returns a dictionary of DatasetDicts which is not compatible with other datasets but is faster
            return combined_data

class CombinedNewsTitleDataset(TRNewsDataset):
    DATASET_NAME = "combined_news"
    DATASET_INFO = ["tr_news", "mlsum"]

    def load_dataset(self, split=None):
        trnews = TRNewsDataset().load_dataset(split)
        mlsum = MLSumDataset().load_dataset(split)
        mlsum = mlsum.rename_column("text", "content")
        if split is not None:
            return datasets.concatenate_datasets([trnews, mlsum])
        else:
            combined_data = {}
            for key in trnews.keys():
                combined_data[key] = datasets.concatenate_datasets([trnews[key], mlsum[key]])
            # Returns DatasetDict object which is compatible with other datasets but takes a lot of time
            # return datasets.Dataset.from_dict(combined_data)
            # Returns a dictionary of DatasetDicts which is not compatible with other datasets but is faster
            return combined_data

class OpenSubtitlesDataset(BaseDataset):
    DATASET_NAME = "opensubtitles"
    DATASET_INFO = "mrbesher/tr-paraphrase-opensubtitles2018"

    def preprocess_data(self, examples):
        return {"input_text": examples["src"], "target_text": examples["tgt"]}

class TatoebaDataset(BaseDataset):
    DATASET_NAME = "tatoeba"
    DATASET_INFO = "mrbesher/tr-paraphrase-tatoeba"

    def preprocess_data(self, examples):
        return {"input_text": examples["src"], "target_text": examples["tgt"]}

class TEDDataset(BaseDataset):
    DATASET_NAME = "ted"
    DATASET_INFO = "mrbesher/tr-paraphrase-ted2013"

    def preprocess_data(self, examples):
        return {"input_text": examples["src"], "target_text": examples["tgt"]}

class LocalDataset(BaseDataset):

    def __init__(self, dataset_loc):
        super().__init__()
        self.dataset_loc = dataset_loc

    def load_dataset(self, split=None, **kwargs):
        return datasets.load_dataset(self.dataset_loc, data_files=self.dataset_info, split=split, **kwargs)


class STSb_TRDataset(LocalDataset):
    DATASET_NAME = "stsb_tr"
    DATASET_INFO = {'train': 'stsb_tr_train.tsv', 'test': 'stsb_tr_test.tsv', 'validation': 'stsb_tr_dev.tsv'}

    def preprocess_data(self, examples, skip_output_processing=False):
        input = [f"ilk cümle: {examples['sentence1'][i]} ikinci cümle: {examples['sentence2'][i]}" for i in range(len(examples["sentence1"]))]
        # If used with the classification mode, skip the output processing
        if skip_output_processing:
            return {"input_text": input, "label": examples["score"]}
        output = [str(ex) for ex in examples["score"]]
        return {"input_text": input, "target_text": output}

    def postprocess_data(self, examples):
        def convert_sts_label(label):
            try:
                return(float(label.strip()))
            except:
                return 0
        return [convert_sts_label(ex) for ex in examples]

class NLI_TRDataset(BaseDataset):
    DATASET_INFO = ("nli_tr", None)
    IN_LABEL_DICT = {0: "gereklilik", 1: "nötr", 2:"çelişki"}
    OUT_LABEL_DICT = {v: k for k, v in IN_LABEL_DICT.items()}
    def __init__(self, dataset_name=None):
        # dataset_name is either "nli_tr", "snli_tr" or "multinli_tr"
        super().__init__(dataset_name)
        self.dataset_info = (self.DATASET_INFO[0], dataset_name)

    def load_dataset(self, split=None):
        if self.dataset_name == "nli_tr":
            if split == "train":
                mnli_tr = NLI_TRDataset("multinli_tr").load_dataset(split)
                snli_tr = NLI_TRDataset("snli_tr").load_dataset(split)
                dataset = datasets.concatenate_datasets([mnli_tr, snli_tr])
            else:
                dataset = NLI_TRDataset("snli_tr").load_dataset(split)
        elif self.dataset_name == 'snli_tr':
            dataset = super().load_dataset(split)
            """elif split is None:
                snli_tr["train"] = snli_tr["train"].filter(lambda example: example["label"] != -1)"""
        elif self.dataset_name == 'multinli_tr' and split == "test":
            dataset = super().load_dataset(split="validation_mismatched")
        else:
            dataset = super().load_dataset(split)
        return dataset.filter(lambda example: example["label"] != -1)

    def preprocess_data(self, examples, skip_output_processing=False):

        input = [f"hipotez: {examples['hypothesis'][i]} önerme: {examples['premise'][i]}" for i in range(len(examples["premise"]))]
        # If used with the classification mode, skip the output processing
        if skip_output_processing:
            return {"input_text": input, "label": examples["label"]}
        output = [NLI_TRDataset.IN_LABEL_DICT[ex] for ex in examples["label"]]
        return {"input_text": input, "target_text": output}

    def postprocess_data(self, examples):
        return [NLI_TRDataset.OUT_LABEL_DICT.get(ex.strip(), -1) for ex in examples]

class QADataset(BaseDataset):
    DATASET_NAME = "qa"

    def postprocess_data(self, examples):
        return [ex.strip() for ex in examples]

class ExamsDataset(QADataset):
    DATASET_NAME = "exams"
    DATASET_INFO = ("exams", "crosslingual_tr")

    def load_dataset(self, split=None):
        if split == 'test':
            # Exams dataset doesn't have a test set, so we use the validation set as test set
            return super().load_dataset(split='validation')
        else:
            return super().load_dataset(split)

    def preprocess_data(self, examples):
        input_texts, target_texts = [], []
        for question, answer_key in zip(examples["question"], examples["answerKey"]):
            question_str = question["stem"]
            choices = question["choices"]
            if answer_key not in choices['label']:
                input_texts.append(question_str)
                target_texts.append('')
                continue
            answer_order = choices['label'].index(answer_key)
            answer = choices['text'][answer_order]
            if not answer:
                continue
            input_texts.append(question_str)
            target_texts.append(answer)
        return {"input_text": input_texts, 'target_text': target_texts}

class TQUADDataset(LocalDataset, QADataset):
    DATASET_NAME = "tquad"
    DATASET_INFO = {'train': 'train-v0.1.json', 'test': 'dev-v0.1.json'}

    def load_dataset(self, split=None):
        return super().load_dataset(split, field='data')

    def preprocess_data(self, examples):
        input_texts, target_texts = [], []
        for paragraphs in examples['paragraphs']:
            for paragraph in paragraphs:
                qas = paragraph['qas']
                context = paragraph['context'].strip()
                for qa in qas:
                    question = qa['question'].strip()
                    answers = qa['answers']
                    answer = answers[0]['text'].strip()
                    input_text = f"Bağlam: {context} | Soru: {question}"
                    target_text = answer
                    input_texts.append(input_text)
                    target_texts.append(target_text)
        return {"input_text": input_texts, "target_text": target_texts}

class MKQADataset(QADataset):
    DATASET_NAME = "mkqa"
    DATASET_INFO = "mkqa"

    def load_dataset(self, split='train'):
        return datasets.load_dataset('mkqa', split='train')

    def preprocess_data(self, examples):
        input_texts, target_texts = [], []
        for queries, answers in zip(examples['queries'], examples['answers']):
            query = queries['tr']
            answer = answers['tr'][0]['text']
            if not answer:
                input_texts.append(query)
                target_texts.append('')
                continue
            input_texts.append(query)
            target_texts.append(answer)
        return {"input_text": input_texts, "target_text": target_texts}

class NERDataset(BaseDataset):
    NER_label_translation_d = {"Kişi": "PER", "Yer": "LOC", "Kuruluş": "ORG"}
    NER_label_int_dict = {"PER": 1, "LOC": 3, "ORG": 5}
    BIO_mapping = {
        "O": 0,
        "B-PERSON": 1,
        "I-PERSON": 2,
        "B-LOCATION": 3,
        "I-LOCATION": 4,
        "B-ORGANIZATION": 5,
        "I-ORGANIZATION": 6,
    }
    label_mapping = {
        0: "O",
        1: "B-PERSON",
        2: "I-PERSON",
        3: "B-LOCATION",
        4: "I-LOCATION",
        5: "B-ORGANIZATION",
        6: "I-ORGANIZATION",
    }

    def preprocess_data(self, examples, tokenizer):
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
        inputs = []
        labels = []
        for i, label in enumerate(examples[f"ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(NERDataset.BIO_mapping[label[word_idx]] if isinstance(label[word_idx], str) else label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
            inputs.append(" ".join(examples["tokens"][i]).strip())
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def postprocess_labels(self, examples):
        preds, labels = examples
        true_predictions = [
            [NERDataset.label_mapping[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(preds, labels)
        ]
        true_labels = [
            [NERDataset.label_mapping[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(preds, labels)
        ]
        return true_predictions, true_labels

    def postprocess_text(self, examples, inputs):
        labels = []
        for example, input_t in zip(examples, inputs):
            example = example.strip()
            input_tokens = input_t.split(' ')
            label_l = ['O' for _ in range(len(input_tokens))]
            if example == 'Bulunamadı.':
                labels.append(label_l)
            else:
                type_split = example.split(' | ')
                for type_el in type_split:
                    if ': ' not in type_el:
                        continue
                    el_split = type_el.split(': ')
                    tag_type = el_split[0].strip()
                    if tag_type not in NERDataset.NER_label_translation_d:
                        continue
                    if ', ' not in el_split[1]:
                        el_l = [el_split[1]]
                    else:
                        el_l = el_split[1].split(', ')
                    for el in el_l:
                        if el.strip() == '':
                            continue
                        el_split = el.split(' ')
                        if el_split[0] not in input_tokens or el_split[-1] not in input_tokens:
                            continue
                        if len(el_split) == 1:
                            start = input_tokens.index(el_split[0])
                            label_l[start] = 'B-' + NERDataset.NER_label_translation_d[tag_type]
                        else:
                            start = input_tokens.index(el_split[0])
                            label_l[start] = 'B-' + NERDataset.NER_label_translation_d[tag_type]
                            end = input_tokens.index(el_split[-1])
                            for i in range(start+1, end+1):
                                label_l[i] = 'I-' + NERDataset.NER_label_translation_d[tag_type]
                labels.append(label_l)
        return labels

    def postprocess_data(self, examples, inputs=None):
        if inputs is None:
           return self.postprocess_labels(examples)
        else:
            return self.postprocess_text(examples, inputs)

class WikiANNDataset(NERDataset):
    DATASET_NAME = "wikiann"
    DATASET_INFO = ("wikiann", "tr")

    def preprocess_data(self, examples, skip_output_processing=False, tokenizer=None):
        if skip_output_processing:
            return super().preprocess_data(examples, tokenizer)
        input_texts = []
        target_texts = []
        for tokens, spans in zip(examples['tokens'], examples['spans']):
            tag_type = ''
            tag_dict = {}
            for span in spans:
                span = span.replace('PER: ', 'Kişi: ').replace('LOC: ', 'Yer: ').replace('ORG: ', 'Kuruluş: ')
                if span.startswith('Kişi: '):
                    tag_type = 'PERSON'
                elif span.startswith('Yer: '):
                    tag_type = 'LOCATION'
                elif span.startswith('Kuruluş: '):
                    tag_type = 'ORGANIZATION'
                if tag_type not in tag_dict:
                    tag_dict[tag_type] = []
                tag_dict[tag_type].append(span.replace('Kişi: ', '').replace('Yer: ', '').replace('Kuruluş: ', ''))
            for tag_type in tag_dict.keys():
                new_l = []
                for el in tag_dict[tag_type]:
                    if el not in new_l:
                        new_l.append(el)
                tag_dict[tag_type] = new_l
            input_text = ' '.join(tokens)
            target_l = []
            target_text = ''
            for tag_type in tag_dict.keys():
                target_l.append(f'{tag_type}: {", ".join(tag_dict[tag_type])}')
            target_text = ' | '.join(target_l)
            target_text = target_text.replace('PERSON: ', 'Kişi: ').replace('LOCATION: ', 'Yer: ').replace('ORGANIZATION: ', 'Kuruluş: ').strip()
            input_text = input_text.strip()
            if not target_text:
                target_text = 'Bulunamadı.'
            input_texts.append(input_text)
            target_texts.append(target_text)
        return {'input_text': input_texts, 'target_text': target_texts}

class MilliyetNERDataset(LocalDataset,NERDataset):
    DATASET_NAME = "milliyet_ner"
    DATASET_INFO = {'train': 'train.json', 'test': 'test.json', 'validation': 'dev.json'}

    def __init__(self, dataset_loc):
        super().__init__(dataset_loc)

    def load_dataset(self, split=None):
        for _, filename in self.dataset_info.items():
            data_file = Path(self.dataset_loc) / filename
            if data_file.exists():
                continue
            else:
                with open(data_file.with_suffix('.txt'), 'r', encoding='utf-8') as f:
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
                    el = {'tokens': tokens, 'ner_tags': tags}
                    with open(data_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(el) + '\n')
        return super().load_dataset(split)

    def preprocess_data(self, examples, skip_output_processing=False, tokenizer=None):
        if skip_output_processing:
            return super().preprocess_data(examples, tokenizer)
        input_texts, target_texts = [], []
        for tokens, tags in zip(examples['tokens'], examples['ner_tags']):
            token_str, tag_type = '', ''
            tag_dict = {}
            for j, tag in enumerate(tags):
                if tag == 'O':
                    if token_str:
                        if tag_type not in tag_dict:
                            tag_dict[tag_type] = []
                        tag_dict[tag_type].append(token_str)
                    token_str, tag_type = '', ''
                elif tag.startswith('B-'):
                    if token_str:
                        if tag_type not in tag_dict:
                            tag_dict[tag_type] = []
                        tag_dict[tag_type].append(token_str)
                    tag_type = tag[2:]
                    token_str = tokens[j]
                elif tag.startswith('I-'):
                    token_str += ' ' + tokens[tags.index(tag)]
            if token_str:
                if tag_type not in tag_dict:
                    tag_dict[tag_type] = []
                tag_dict[tag_type].append(token_str)
            for j, tag_type in enumerate(tag_dict):
                new_l = []
                for el in tag_dict[tag_type]:
                    if el not in new_l:
                        new_l.append(el)
                tag_dict[tag_type] = new_l
            input_text = ' '.join(tokens)
            target_l = []
            target_text = ''
            for j, tag_type in enumerate(tag_dict):
                target_l.append(f'{tag_type}: {", ".join(tag_dict[tag_type])}')
            target_text = ' | '.join(target_l)
            input_text = input_text.strip()
            target_text = target_text.replace('PERSON: ', 'Kişi: ').replace('LOCATION: ', 'Yer: ').replace('ORGANIZATION: ', 'Kuruluş: ').strip()
            if not target_text:
                target_text = 'Bulunamadı.'
            input_texts.append(input_text)
            target_texts.append(target_text)
        return {'input_text': input_texts, 'target_text': target_texts}

class POSDataset(LocalDataset):
    DATASET_NAME = "pos"
    DATASET_INFO = {'train': 'train.json', 'test': 'test.json', 'validation': 'dev.json'}
    POS_TR_DICT = { "ADP": "edat", "AUX": "yardımcı", "PRON": "zamir", "NOUN": "isim", "PROPN": "özel", "INTJ": "ünlem", "PART": "tanımcık", "CCONJ": "eşgüdümlü", "VERB": "fiil", "SYM": "sembol", "DET": "belirteç", "ADV": "zarf", "ADJ": "sıfat", "X": "diğer", "SCONJ": "yantümce", "NUM": "sayı", "PUNCT": "noktalama" }
    POS_INT_DICT = {"edat": 0, "yardımcı": 1, "zamir": 2, "isim": 3, "özel": 4, "ünlem": 5, "tanımcık": 6, "eşgüdümlü": 7, "fiil": 8, "sembol": 9, "belirteç": 10, "zarf": 11, "sıfat": 12, "diğer": 13, "yantümce": 14, "sayı": 15, "noktalama": 16}
    label_mapping = {v: k for k, v in POS_INT_DICT.items()}

    def __init__(self, dataset_loc=None, dataset_raw_info=None):
        super().__init__(dataset_loc)
        self.DATASET_RAW_INFO = dataset_raw_info

    def load_dataset(self, split=None):
        md_pattern = re.compile('^# (.+?) = (.+?)$')
        annotation_pattern = re.compile('(.+\t){9}.+')
        for split_t, filename in self.DATASET_RAW_INFO.items():
            data_file = Path(self.dataset_loc) / filename
            output_file = Path(self.dataset_loc) / self.DATASET_INFO[split_t]
            if output_file.exists():
                continue
            else:
                with open(data_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                sents = content.split('\n\n')
                for sent in sents:
                    lines = sent.split('\n')
                    sent_id = ''
                    d_t = {}
                    id_l, token_l, tag_l = [], [], []
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
                            for row in annotation.split('\n'):
                                if row.strip() == '':
                                    break
                                fields = row.split('\t')
                                id_t, token, tag = fields[0], fields[1], fields[3]
                                if '-' in id_t:
                                    continue
                                id_l.append(id_t)
                                token_l.append(token)
                                tag_l.append(tag)
                            d_t['split'] = split_t
                            d_t['tokens'] = token_l
                            d_t['tags'] = tag_l
                            d_t['sent_id'] = sent_id
                            d_t['ids'] = id_l
                            with open(output_file, 'a', encoding='utf-8') as f:
                                f.write(json.dumps(d_t) + '\n')
                            break
        return super().load_dataset(split)

    def preprocess_labels(self, examples, tokenizer):
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
        inputs = []
        labels = []
        for i, label in enumerate(examples[f"tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(POSDataset.POS_INT_DICT[POSDataset.POS_TR_DICT[label[word_idx]]] if isinstance(label[word_idx], str) else label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
            inputs.append(" ".join(examples["tokens"][i]).strip())
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def preprocess_data(self, examples, skip_output_processing=False, tokenizer=None):
        if skip_output_processing:
            return self.preprocess_labels(examples, tokenizer)
        input_texts, target_texts = [], []
        for ids, tokens, tags in zip(examples['ids'], examples['tokens'], examples['tags']):
            tag_l = []
            split_token = 0
            for id_t, form, pos in zip(ids, tokens, tags):
                if '-' in id_t:
                    split_token = 2
                if pos == '_':
                    continue
                if split_token == 1:
                    tag_l.append('-{}/{}'.format(form, POSDataset.POS_TR_DICT[pos]))
                else:
                    tag_l.append('{}/{}'.format(form, POSDataset.POS_TR_DICT[pos]))
                if split_token != 0:
                    split_token -= 1
            output = ' '.join(tag_l)
            input_texts.append(' '.join(tokens))
            target_texts.append(output)
        return {"input_text": input_texts, "target_text": target_texts}

    def postprocess_labels(self, examples):
        preds, labels = examples
        true_predictions = [
            [POSDataset.label_mapping[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(preds, labels)
        ]
        true_labels = [
            [POSDataset.label_mapping[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(preds, labels)
        ]
        return true_predictions, true_labels

    def postprocess_text(self, examples, inputs):
        labels = []
        for input_t, example in zip(inputs, examples):
            example = example.strip()
            input_tokens = input_t.split(' ')
            tokens = example.split(' ')
            label_l = ['O' for i in range(len(input_tokens))]
            for i, token in enumerate(tokens):
                if i >= len(label_l):
                    break
                token_split = token.split('/')
                label = token_split[-1].strip()
                if label != '':
                    label_l[i] = 'B-' + label
            labels.append(label_l)
        return labels

    def postprocess_data(self, examples, inputs=None):
        if inputs is None:
           return self.postprocess_labels(examples)
        else:
            return self.postprocess_text(examples, inputs)

class UDBOUNDataset(POSDataset):
    DATASET_NAME = "boun"
    DATASET_RAW_INFO =  {'train': 'tr_boun-ud-train.conllu', 'test': 'tr_boun-ud-test.conllu', 'validation': 'tr_boun-ud-dev.conllu'}

    def __init__(self, dataset_loc=None):
        super().__init__(dataset_loc, self.DATASET_RAW_INFO)

class UDIMSTDataset(POSDataset):
    DATASET_NAME = "imst"
    DATASET_RAW_INFO =  {'train': 'tr_imst-ud-train.conllu', 'test': 'tr_imst-ud-test.conllu', 'validation': 'tr_imst-ud-dev.conllu'}

    def __init__(self, dataset_loc=None):
        super().__init__(dataset_loc, self.DATASET_RAW_INFO)

class ClassificationDataset(BaseDataset):
    IN_LABEL_DICT = None
    OUT_LABEL_DICT = None

    def __init__(self, dataset_name=None):
        super().__init__(dataset_name)
        self.OUT_LABEL_DICT = {v: k for k, v in self.IN_LABEL_DICT.items()}

    def postprocess_data(self, examples):
        return [self.OUT_LABEL_DICT.get(ex.strip(), -1) for ex in examples]

    def load_dataset(self, split=None):
        return super().load_dataset(split)

class TTC4900Dataset(ClassificationDataset):
    DATASET_NAME = "ttc4900"
    DATASET_INFO = "ttc4900"
    IN_LABEL_DICT = {0: "siyaset", 1: "dünya", 2: "ekonomi", 3: "kültür", 4: "sağlık", 5: "spor", 6: "teknoloji"}

    def __init__(self, dataset_name=None):
        super().__init__(dataset_name)

    def preprocess_data(self, examples, skip_output_processing=False):
        # If used with the classification mode, don't process the output
        if skip_output_processing:
            return {"input_text": examples["text"], "label": examples["category"]}
        output = [self.IN_LABEL_DICT[ex] for ex in examples["category"]]
        return {"input_text": examples["text"], "target_text": output}

    def load_dataset(self, split=None):
        dataset = super().load_dataset(split)
        print("Deduplicating data")
        return super().deduplicate_data(dataset, "text")

class ProductDataset(ClassificationDataset):
    DATASET_NAME = "turkish_product_reviews"
    DATASET_INFO = "turkish_product_reviews"
    IN_LABEL_DICT = {0: "negatif", 1: "pozitif"}

    def __init__(self, dataset_name=None):
        super().__init__(dataset_name)

    def preprocess_data(self, examples, skip_output_processing=False):
        # If used with the classification mode, don't process the output
        if skip_output_processing:
            return {"input_text": examples["sentence"], "label": examples["sentiment"]}
        output = [self.IN_LABEL_DICT[ex] for ex in examples["sentiment"]]
        return {"input_text": examples["sentence"], "target_text": output}

    def load_dataset(self, split=None):
        dataset = super().load_dataset(split)
        print("Deduplicating data")
        return super().deduplicate_data(dataset, "sentence")

class SentimentTweetDataset(ClassificationDataset):
    DATASET_NAME = "sentiment_tweet"
    DATASET_INFO = {'train': 'formatted_train.csv', 'test': 'formatted_test.csv'}
    IN_LABEL_DICT = {0: "olumsuz", 1: "nötr", 2: "olumlu"}

    def __init__(self, dataset_loc):
        super().__init__(self)
        self.dataset_loc = dataset_loc

    def preprocess_data(self, examples, skip_output_processing=False):
        # If used with the classification mode, don't process the output
        if skip_output_processing:
            return {"input_text": examples["text"], "label": examples["label"]}
        output = [self.IN_LABEL_DICT[ex] for ex in examples["label"]]
        return {"input_text": examples["text"], "target_text": output}

    def postprocess_data(self, examples):
        return [self.OUT_LABEL_DICT.get(ex.strip(), -1) for ex in examples]

    def load_dataset(self, split=None):
        dataset = LocalDataset.load_dataset(self, split)
        #dataset = datasets.load_dataset(self.dataset_loc, data_files=self.dataset_info, split=split)
        print("Deduplicating data")
        return super().deduplicate_data(dataset, "text")

DATASET_MAPPING_NAMES = [
        ("tr_news", "TRNewsDataset"),
        ("tr_news_title", "TRNewsTitleDataset"),
        ("mlsum", "MLSumDataset"),
        ("mlsum_title", "MLSumTitleDataset"),
        ("combined_news", "CombinedNewsDataset"),
        ("combined_news_title", "CombinedNewsTitleDataset"),
        ("opensubtitles", "OpenSubtitlesDataset"),
        ("tatoeba", "TatoebaDataset"),
        ("ted", "TEDDataset"),
        ("nli_tr", "NLI_TRDataset"),
        ("snli_tr", "NLI_TRDataset"),
        ("multinli_tr", "NLI_TRDataset"),
        ("exams", "ExamsDataset"),
        ("tquad", "TQUADDataset"),
        ("mkqa", "MKQADataset"),
        ("wikiann", "WikiANNDataset"),
        ("milliyet", "MilliyetNERDataset"),
        ("boun", "UDBOUNDataset"),
        ("imst", "UDIMSTDataset"),
        ("stsb_tr", "STSb_TRDataset"),
        ("ttc4900", "TTC4900Dataset"),
        ("tr_product_reviews", "ProductDataset"),
        ("17bintweet_sentiment", "SentimentTweetDataset"),
    ]

def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)

def initialize_dataset(dataset_name, dataset_loc=None):
    for dataset_mapping_name in DATASET_MAPPING_NAMES:
        if dataset_name == dataset_mapping_name[0]:
            dataset_class = str_to_class(dataset_mapping_name[1])
            if dataset_loc != "" and dataset_loc is not None:
                dataset = dataset_class(dataset_loc)
            else:
                dataset = dataset_class(dataset_name)
            return dataset
    raise NotImplementedError
