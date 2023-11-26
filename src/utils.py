
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

