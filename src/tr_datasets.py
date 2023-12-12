import datasets 

DATASET_MAPPING_NAMES = [
        ("tr_news", "TRNewsDataset"),
        ("mlsum", "MLSumDataset"),
        ("combined_news", "CombinedNewsDataset"),
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
        ("milliyet_ner", "MilliyetNERDataset"),
        ("boun", "UDBOUNDataset"),
        ("imst", "UDIMSTDataset"),
    ]

def initialize_dataset(dataset_name, dataset_loc=None):
    for dataset_mapping_name in DATASET_MAPPING_NAMES:
        if dataset_name == dataset_mapping_name[0]:
            dataset_class = dataset_mapping_name[1]
            if dataset_loc is not None:
                dataset = dataset_class(dataset_loc)
            else:
                dataset = dataset_class()
            return dataset
    raise NotImplementedError

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
        return [ex.strip() for ex in examples]

class TRNewsDataset(BaseDataset): 
    DATASET_NAME = "tr_news"
    DATASET_INFO = "batubayk/TR-News"
    
    def preprocess_data(self, examples):
        return {"input_text": examples["content"], "target_text": examples["title"]}

class MLSumDataset(BaseDataset):
    DATASET_NAME = "mlsum"
    DATASET_INFO = ("mlsum", "tu")
    
    def preprocess_data(self, examples):
        return {"input_text": examples["text"], "target_text": examples["summary"]}
    
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
    
    def preprocess_data(self, examples):
        input = [f"ilk cümle: {examples['sentence1'][i]} ikinci cümle: {examples['sentence2'][i]}" for i in range(len(examples["sentence1"]))]
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
    NLI_LABEL_DICT = {0: "gereklilik", 1: "nötr", 2:"çelişki"}
    def __init__(self, dataset_name=None):
        # dataset_name is either "nli_tr", "snli_tr" or "multinli_tr"
        super().__init__(dataset_name)
        self.dataset_info = self.DATASET_INFO[0], dataset_name
    
    def load_dataset(self, split=None):
        if self.dataset_name == "nli_tr":
            mnli_tr = NLI_TRDataset("multinli_tr").load_dataset(split)
            snli_tr = NLI_TRDataset("snli_tr").load_dataset(split)
            if split is not None:
                return datasets.concatenate_datasets([mnli_tr, snli_tr]) 
            else:
                combined_data = {}
                for key in snli_tr.keys():
                    combined_data[key] = datasets.concatenate_datasets([snli_tr[key], mnli_tr[key]]) 
                # Returns DatasetDict object which is compatible with other datasets but takes a lot of time
                # return datasets.Dataset.from_dict(combined_data)
                # Returns a dictionary of DatasetDicts which is not compatible with other datasets but is faster
                return combined_data 
        elif self.dataset_name == 'snli_tr':
            snli_tr = NLI_TRDataset("snli_tr").load_dataset(split)
            if split == 'train' or split is None:
                snli_tr["train"] = snli_tr["train"].filter(lambda example: example["label"] != -1)
            return snli_tr
        else:
            return super().load_dataset(split)
        
    def preprocess_data(self, examples, skip_output_processing=False):
        
        input = [f"hipotez: {examples['hypothesis'][i]} önerme: {examples['premise'][i]}" for i in range(len(examples["premise"]))]
        # If used with the classification mode, skip the output processing
        if skip_output_processing:
            return {"input_text": input, "label": examples["label"]}
        output = [NLI_TRDataset.NLI_LABEL_DICT[ex] for ex in examples["label"]]
        return {"input_text": input, "target_text": output}
    
    def postprocess_data(self, examples):
        return [NLI_TRDataset.NLI_LABEL_DICT.get(ex.strip(), -1) for ex in examples]

class ExamsDataset(BaseDataset):
    DATASET_NAME = "exams"
    DATASET_INFO = ("exams", "crosslingual_tr")

    def load_dataset(self, split=None):
        if split == 'test':
            # Exams dataset doesn't have a test set, so we use the validation set as test set
            return super().load_dataset(split='validation')
        else:
            return super().load_dataset(split)
        
    def preprocess_data(self, examples, task='qa'):
        return self.preprocess_qa(examples) if task == 'qa' else self.preprocess_qg(examples)
    
    def preprocess_question_answering(self, examples):
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
    
    def preprocess_question_generation(self, examples):
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
        return {"input_text": target_texts, 'target_text': input_texts}

class TQUADDataset(LocalDataset):
    DATASET_NAME = "tquad"
    DATASET_INFO = {'train': 'train-v0.1.json', 'test': 'dev-v0.1.json'}

    def load_dataset(self, split=None):
        return super().load_dataset(split, field='data')
    
    def preprocess_data(self, examples, task='qa'):
        return self.preprocess_question_answering(examples) if task == 'qa' else self.preprocess_question_generation(examples)
    
    def preprocess_question_answering(self, examples):
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
    
    def preprocess_question_generation(self, examples):
        input_texts, target_texts = [], []
        for paragraphs in examples['paragraphs']:
            for paragraph in paragraphs:
                qas = paragraph['qas']
                context = paragraph['context'].strip()
                for qa in qas:
                    question = qa['question'].strip()
                    answers = qa['answers']
                    answer = answers[0]['text'].strip()
                    input_text = f"Bağlam: {context} | Cevap: {answer}"
                    target_text = question
                    input_texts.append(input_text)
                    target_texts.append(target_text)
        return {"input_text": input_texts, "target_text": target_texts}
    
class MKQADataset(BaseDataset):
    DATASET_NAME = "mkqa"
    DATASET_INFO = "mkqa"    

    def preprocess_data(self, examples, task='qa'):
        return self.preprocess_question_answering(examples) if task == 'qa' else self.preprocess_question_generation(examples)
    
    def preprocess_question_answering(self, examples):
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

    def preprocess_question_generation(self, examples):
        input_texts, target_texts = [], []
        for queries, answers in zip(examples['queries'], examples['answers']):
            query = queries['tr']
            answer = answers['tr'][0]['text']
            if not answer:
                input_texts.append(answer)
                target_texts.append('')
                continue
            input_texts.append(answer)
            target_texts.append(query)
        return {"input_text": input_texts, "target_text": target_texts}

class WikiANNDataset(BaseDataset):
    DATASET_NAME = "wikiann"
    DATASET_INFO = ("wikiann", "tr")

    def preprocess_data(self, examples):
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
    
class MilliyetNERDataset(BaseDataset):
    DATASET_NAME = "milliyet_ner"
    DATASET_INFO = {'train': 'train.txt', 'test': 'test.txt', 'validation': 'dev.txt'}

    def load_dataset(self, split=None):
        pass 

    def preprocess_data(self, examples):
        input_texts, target_texts = [], []
        for tokens, tags in zip(examples['tokens'], examples['tags']):
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
    
class POSDataset(BaseDataset):
    DATASET_NAME = "pos"
    DATASET_INFO = "udpos"

    def load_dataset(self, split=None):
        pass
    
    def preprocess_data(self, examples):
        pos_d_tr = { "ADP": "edat", "AUX": "yardımcı", "PRON": "zamir", "NOUN": "isim", "PROPN": "özel", "INTJ": "ünlem", "PART": "tanımcık", "CCONJ": "eşgüdümlü", "VERB": "fiil", "SYM": "sembol", "DET": "belirteç", "ADV": "zarf", "ADJ": "sıfat", "X": "diğer", "SCONJ": "yantümce", "NUM": "sayı", "PUNCT": "noktalama" }
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
                        tag_l.append('-{}/{}'.format(form, pos_d_tr[pos]))
                    else:
                        tag_l.append('{}/{}'.format(form, pos_d_tr[pos]))
                    if split_token != 0:
                        split_token -= 1
            output = ' '.join(tag_l)
            input_texts.append(' '.join(tokens))
            target_texts.append(output)
        return {"input_text": input_texts, "target_text": target_texts}
    
class UDBOUNDataset(POSDataset):
    DATASET_NAME = "boun"
    DATASET_INFO =  {'train': 'tr_boun-ud-train.conllu', 'test': 'tr_boun-ud-test.conllu', 'validation': 'tr_boun-ud-dev.conllu'}

class UDIMSTDataset(POSDataset):
    DATASET_NAME = "imst"
    DATASET_INFO =  {'train': 'tr_imst-ud-train.conllu', 'test': 'tr_imst-ud-test.conllu', 'validation': 'tr_imst-ud-dev.conllu'}
