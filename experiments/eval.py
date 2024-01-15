import hydra
import json
from pathlib import Path
import os 
from omegaconf import DictConfig
from turkish_lm_tuner.dataset_processor import DatasetProcessor
from turkish_lm_tuner.evaluator import EvaluatorForConditionalGeneration, EvaluatorForClassification

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
formatter = logging.Formatter('%(levelname)s - %(asctime)s - %(name)s: %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

@hydra.main(config_path="generation_conf", config_name="default")
def main(cfg: DictConfig):
    model_path = cfg.model_path
    tokenizer_path = cfg.tokenizer_path
    dataset_name = cfg.dataset_name
    task = cfg.task
    task_format = cfg.task_format
    task_mode = cfg.task_mode
    max_input_length = cfg.max_input_length
    max_target_length = cfg.max_target_length
    test_params = cfg.test_params
    generation_params = cfg.generation_params
    dataset_location = cfg.dataset_loc

    logger.info("Loading test dataset")
    dataset_processor = DatasetProcessor(dataset_name, task, task_format, task_mode, tokenizer_path, max_input_length, max_target_length, dataset_location)
    
    try:
        test_dataset = dataset_processor.load_and_preprocess_data(split="test")  # Use split="test[:10]" to test for small sample
        test_exists = True
        val_exists = True
    except:
        test_exists = False
        logger.info("No existing test split!")
        try: 
            eval_dataset = dataset_processor.load_and_preprocess_data(split='validation')
            val_exists = True
        except ValueError:
            logger.info("No existing validation split!")
            val_exists = False
    
    if not val_exists and not test_exists:
        train_set = dataset_processor.load_and_preprocess_data(split='train')
        logger.info("Creating random train, validation and test splits")
        train_set = train_set.train_test_split(test_size=0.2, seed=25)
        _, eval_test_dataset = train_set["train"], train_set["test"] 
        eval_test_dataset = eval_test_dataset.train_test_split(test_size=0.5, seed=25)
        _, test_dataset = eval_test_dataset["train"], eval_test_dataset["test"] 
    elif not test_exists:
        train_set = dataset_processor.load_and_preprocess_data(split='train')
        logger.info("Creating random train, test splits")
        train_set = train_set.train_test_split(test_size=0.1, seed=25)
        _, test_dataset = train_set["train"], train_set["test"] 

    postprocess_fn = dataset_processor.dataset.postprocess_data

    logger.info("test_dataset[0]: %s", test_dataset[0])
    logger.info("test_dataset: %s", test_dataset)
    

    if task_format == 'conditional_generation':
        logger.info("Evaluating in conditional generation mode")
        evaluator = EvaluatorForConditionalGeneration(model_path, tokenizer_path, task, max_input_length, max_target_length, test_params, generation_params, postprocess_fn)
    elif task_format == 'classification':
        logger.info("Evaluating in classification mode")
        evaluator = EvaluatorForClassification(model_path, tokenizer_path, task, test_params)

  
    logger.info("Evaluating model")
    results = evaluator.evaluate_model(test_dataset)
    logger.info("Result: %s", results)    
    json.dump(results.metrics, open(os.path.join(test_params['output_dir'], "results.json"), "w"))


if __name__ == "__main__":
    main()
