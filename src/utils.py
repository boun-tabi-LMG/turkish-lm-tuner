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
    return {"input_text": f"hipotez: {examples['hypothesis']} önerme: {examples['premise']}"}

def preprocess_exams_qa(examples):
    question = examples["question"]
    question_str = question["stem"]
    answer_key = examples["answerKey"]
    choices = question["choices"]
    try:
        answer_order = choices['label'].index(answer_key) # won't work probably
    except:
        return None
    answer = choices['text'][answer_order]
    if not answer:
        return None
    return {"input_text": question_str, "target_text": answer}

def preprocess_exams_qg(examples):
    question = examples["question"]
    question_str = question["stem"]
    answer_key = examples["answerKey"]
    choices = question["choices"]
    try:
        answer_order = choices['label'].index(answer_key) # won't work probably
    except:
        return None
    answer = choices['text'][answer_order]
    if not answer:
        return None
    return {"input_text": answer, "target_text": question_str}

def preprocess_xquad_qa(examples):
    question, context, answers = examples["question"], examples["context"], examples["answers"]
    input_text = f"Bağlam: {context} | Soru: {question}"
    answer = answers["text"][0] # does this work?
    return {"input_text": input_text, "target_text": answer}

def preprocess_xquad_qg(examples):
    question, context, answers = examples["question"], examples["context"], examples["answers"]
    input_text = f"Bağlam: {context} | Cevap: {answers['text'][0]}"
    return {"input_text": input_text, "target_text": question}

def preprocess_mkqa_qa(examples):
    query = examples["queries"]["tr"]
    answer = examples["answers"]["tr"][0]['text']
    if not answer:
        return None
    return {"input_text": query, "target_text": answer}

def preprocess_mkqa_qg(examples):
    query = examples["queries"]["tr"]
    answer = examples["answers"]["tr"][0]['text']
    if not answer:
        return None
    return {"input_text": answer, "target_text": query}

def preprocess_wikiann_ner(examples):
    tokens = examples['tokens']
    spans = examples['spans']
    for i, span in enumerate(spans):
        spans[i] = span.replace('PER: ', 'Kişi: ').replace('LOC: ', 'Yer: ').replace('ORG: ', 'Kuruluş: ')
    tag_type = ''
    tag_dict = {}
    for span in spans:
        if span.startswith('Kişi: '):
            tag_type = 'PERSON'
        elif span.startswith('Yer: '):
            tag_type = 'LOCATION'
        elif span.startswith('Kuruluş: '):
            tag_type = 'ORGANIZATION'
        if tag_type not in tag_dict:
            tag_dict[tag_type] = []
        tag_dict[tag_type].append(span.replace('Kişi: ', '').replace('Yer: ', '').replace('Kuruluş: ', ''))
    for tag_type in tag_dict:
        new_l = []
        for el in tag_dict[tag_type]:
            if el not in new_l:
                new_l.append(el)
        tag_dict[tag_type] = new_l
    input_text = ' '.join(tokens)
    target_l = []
    target_text = ''
    for tag_type in tag_dict:
        target_l.append(f'{tag_type}: {", ".join(tag_dict[tag_type])}')
    target_text = ' | '.join(target_l)
    target_text = target_text.replace('PERSON: ', 'Kişi: ').replace('LOCATION: ', 'Yer: ').replace('ORGANIZATION: ', 'Kuruluş: ').strip()
    input_text = input_text.strip()
    if not target_text:
        target_text = 'Bulunamadı.'
    return {'input_text': input_text, 'target_text': target_text}

def preprocess_xtreme_ner(examples):
    tokens = examples['tokens']
    ner_tags = examples['ner_tags']
    token_str, tag_type = '', ''
    tag_dict = {}
    for j, tag in enumerate(ner_tags):
        if tag == 0:
            if token_str:
                if tag_type not in tag_dict:
                    tag_dict[tag_type] = []
                tag_dict[tag_type].append(token_str)
            token_str, tag_type = '', ''
        elif tag in [1, 3, 5]:
            if token_str:
                if tag_type not in tag_dict:
                    tag_dict[tag_type] = []
                tag_dict[tag_type].append(token_str)
            if tag == 1:
                tag_type = 'PERSON'
            elif tag == 3:
                tag_type = 'ORGANIZATION'
            elif tag == 5:
                tag_type = 'LOCATION'
            token_str = tokens[j]
        elif tag in [2, 4, 6]:
            token_str += ' ' + tokens[j]
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
    return {'input_text': input_text, 'target_text': target_text}
