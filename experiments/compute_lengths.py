from argparse import ArgumentParser
from turkish_lm_tuner.dataset_processor import DatasetProcessor

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--dataset_name', type=str, default='wikiann')
    parser.add_argument('--task', type=str, default='ner')
    parser.add_argument('--dataset_loc', type=str, default='')
    args = parser.parse_args()
    model_name = args.model_name
    dataset_name = args.dataset_name
    print('Dataset:', dataset_name)
    dataset_loc = args.dataset_loc
    task = args.task
    print('Task:', task)
    task_format = 'conditional_generation'
    task_mode = ''
    max_input_length = -1
    max_target_length = -1

    dataset_processor = DatasetProcessor(dataset_name, task, task_format, task_mode, model_name, max_input_length, max_target_length, dataset_loc)
    dataset_processor.load_and_preprocess_data()
