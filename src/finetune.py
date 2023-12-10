from omegaconf import DictConfig
from dataset import DatasetProcessor
from trainer import TrainerForConditionalGeneration, TrainerForClassification

import hydra
import os
# local_rank = int(os.environ["LOCAL_RANK"])


@hydra.main(config_path="../conf", config_name="default")
def main(cfg: DictConfig):
    model_name = cfg.model_name
    dataset_name = cfg.dataset_name
    task = cfg.task
    task_format = cfg.task_format
    task_mode = cfg.task_mode
    max_input_length = cfg.max_input_length
    max_target_length = cfg.max_target_length
    adafactor_scheduler = cfg.adafactor_scheduler
    training_params = cfg.training_params
    dataset_location = cfg.dataset_loc
     
    dataset_processor = DatasetProcessor(dataset_name, task, task_format, task_mode, model_name, max_input_length, max_target_length, dataset_location)
    train_set = dataset_processor.load_and_preprocess_data()
    model_save_path = training_params['output_dir']
    try: 
        eval_dataset = dataset_processor.load_and_preprocess_data(split='validation')
        train_dataset = train_set
    except:
        train_set = train_set.train_test_split(test_size=0.1)
        train_dataset, eval_dataset = train_set["train"], train_set["test"]
    
    test_dataset = dataset_processor.load_and_preprocess_data(split="test")

    print("train", train_dataset)
    print("val", eval_dataset)
    print("test", test_dataset)
    if task_format == 'conditional_generation':
        model_trainer = TrainerForConditionalGeneration(model_name, task, adafactor_scheduler, training_params, model_save_path, dataset_name, max_target_length)
    elif task_format == 'classification':
        model_trainer = TrainerForClassification(model_name, adafactor_scheduler, training_params)

    trainer, model = model_trainer.train_and_evaluate(train_dataset, eval_dataset, test_dataset)

    print("Best model saved at", model_save_path)
    model.save_pretrained(model_save_path)
    # dataset_processor.tokenizer.save_pretrained(model_save_path)

    # If separate evaluation will be made, send the model to the evaluator to avoid re-loading
    # model_trainer.evaluator.evaluate_model(test_dataset, model)

if __name__ == "__main__":
    main()
