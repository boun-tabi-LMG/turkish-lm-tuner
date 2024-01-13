from omegaconf import DictConfig
from turkish_lm_tuner.dataset_processor import DatasetProcessor
from turkish_lm_tuner.trainer import TrainerForConditionalGeneration, TrainerForClassification

import hydra
import os
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
formatter = logging.Formatter('%(levelname)s - %(asctime)s - %(name)s: %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

try:
    local_rank = int(os.environ["LOCAL_RANK"])
except Exception as e:
    logger.info("LOCAL_RANK not found in environment variables. Not using distributed training.")

@hydra.main(config_path="conf", config_name="default")
def main(cfg: DictConfig):
    model_name = cfg.model_name
    dataset_name = cfg.dataset_name
    task = cfg.task
    task_format = cfg.task_format
    task_mode = cfg.task_mode
    max_input_length = cfg.max_input_length
    logger.info(f'Max input length: {max_input_length}')
    max_target_length = cfg.max_target_length
    logger.info(f'Max target length: {max_target_length}')
    training_params = cfg.training_params
    logger.info(f'Training parameters: {training_params}')
    if "optimizer_params" in cfg:
        optimizer_params = cfg.optimizer_params
    else:
        optimizer_params = None
    logger.info(f'Optimizer parameters: {optimizer_params}')

    dataset_location = cfg.dataset_loc
    if "num_labels" in cfg.keys():
        num_labels = cfg.num_labels
     
    dataset_processor = DatasetProcessor(dataset_name, task, task_format, task_mode, model_name, max_input_length, max_target_length, dataset_location)

    train_set = dataset_processor.load_and_preprocess_data(split='train')
    postprocess_fn = dataset_processor.dataset.postprocess_data
    
    model_save_path = training_params['output_dir']
    try:
        logger.info("Loading existing test split")
        test_dataset = dataset_processor.load_and_preprocess_data(split="test")
        train_dataset = train_set
        test_exists = True
    except ValueError:
        logger.info("No existing test split!")
        test_exists = False

    try: 
        logger.info("Loading existing validation split")
        eval_dataset = dataset_processor.load_and_preprocess_data(split='validation')
        train_dataset = train_set
        val_exists = True
    except ValueError:
        logger.info("No existing validation split!")
        val_exists = False

    if not val_exists and not test_exists:
        logger.info("Creating random train, validation and test splits")
        train_set = train_set.train_test_split(test_size=0.2, seed=25)
        train_dataset, eval_test_dataset = train_set["train"], train_set["test"] 
        eval_test_dataset = eval_test_dataset.train_test_split(test_size=0.5, seed=25)
        eval_dataset, test_dataset = eval_test_dataset["train"], eval_test_dataset["test"] 
    elif not val_exists:
        logger.info("Creating random train, validation splits")
        train_set = train_set.train_test_split(test_size=0.1, seed=25)
        train_dataset, eval_dataset = train_set["train"], train_set["test"] 
    elif not test_exists:
        logger.info("Creating random train, test splits")
        train_set = train_set.train_test_split(test_size=0.1, seed=25)
        train_dataset, test_dataset = train_set["train"], train_set["test"] 

    logger.info("Training set size: %d", len(train_dataset))
    logger.info("Validation set size: %d", len(eval_dataset))
    logger.info("Test set size: %d", len(test_dataset))
    logger.info("Training set example: %s", train_dataset[0])
    logger.info("Validation set example: %s", eval_dataset[0])
    logger.info("Test set example: %s", test_dataset[0])

    if task_format == 'conditional_generation':
        logger.info("******Conditional Generation Mode******")
        model_trainer = TrainerForConditionalGeneration(model_name, task, training_params, optimizer_params, model_save_path, max_input_length, max_target_length, postprocess_fn)
    elif task_format == 'classification':
        logger.info("******Classification Mode******")
        model_trainer = TrainerForClassification(model_name, task, training_params, optimizer_params, model_save_path, num_labels, postprocess_fn)

    trainer, model = model_trainer.train_and_evaluate(train_dataset, eval_dataset, test_dataset)

    logger.info("Best model saved at %s", model_save_path)
    model.save_pretrained(model_save_path)
    dataset_processor.tokenizer.save_pretrained(model_save_path)

    # If separate evaluation will be made, send the model to the evaluator to avoid re-loading
    # model_trainer.evaluator.evaluate_model(test_dataset, model)

if __name__ == "__main__":
    main()
