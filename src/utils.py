def default_preprocess_function(examples):
    # Default preprocessing if specific preprocess function is not found
    return {"input_text": examples["text"], "labels": examples["label"]}

def preprocess_trnews_summarization(examples):
    return {"input_text": examples["content"], "target_text": examples["abstract"]}

def preprocess_trnews_title_generation(examples):
    return {"input_text": examples["content"], "target_text": examples["title"]}

def preprocess_paraphrasing(examples):
    return {"input_text": examples["src"], "target_text": examples["tgt"]}

def preprocess_nli(examples):
    nli_label_dict = {0: "gereklilik", 1: "nötr", 2:"çelişki"}
    input = [f"hipotez: {examples['hypothesis'][i]} önerme: {examples['premise'][i]}" for i in range(len(examples["premise"]))]
    output = [nli_label_dict[ex] for ex in examples["label"]]
    return {"input_text": input, "target_text": output}

def preprocess_exams_qa(examples):
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

def preprocess_exams_qg(examples):
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

def preprocess_mkqa_qa(examples):
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

def preprocess_mkqa_qg(examples):
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

def preprocess_wikiann_ner(examples):
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

def preprocess_sts(examples):
    input = [f"ilk cümle: {examples['sentence1'][i]} ikinci cümle: {examples['sentence2'][i]}" for i in range(len(examples["sentence1"]))]
    output = [str(ex) for ex in examples["score"]]
    return {"input_text": input, "target_text": output}

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    print(f"Predictions: {preds[:5]}")
    print(f"Labels: {labels[:5]}")
    return preds, labels

def postprocess_nli(preds, labels):
    nli_label_dict = {"gereklilik":0, "nötr":1 , "çelişki": 2}

    preds = [nli_label_dict.get(pred.strip(), -1) for pred in preds]
    labels = [nli_label_dict.get(label.strip(), -1) for label in labels]
    print(f"Predictions: {preds[:5]}")
    print(f"Labels: {labels[:5]}")
    return preds, labels

def convert_sts_label(label):
    try:
        return(float(label.strip()))
    except:
        return 0

def postprocess_sts(preds, labels):
    preds = [convert_sts_label(pred) for pred in preds]
    labels = [convert_sts_label(label) for label in labels]
    print(f"Predictions: {preds[:5]}")
    print(f"Labels: {labels[:5]}")
    return preds, labels
