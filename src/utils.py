
def preprocess_exams(examples):
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

def preprocess_xquad(examples):
    question, context, answers = examples["question"], examples["context"], examples["answers"]
    input_text = f"Bağlam: {context} | Soru: {question}"
    answer = answers["text"][0] # does this work?
    return {"input_text": input_text, "target_text": answer}

def preprocess_xquad_qg(examples):
    question, context, answers = examples["question"], examples["context"], examples["answers"]
    input_text = f"Bağlam: {context} | Cevap: {answers['text'][0]}"
    return {"input_text": input_text, "target_text": question}

