import os,sys
import yaml
import zipfile
import subprocess
from plantdiseasedetection.logger import logging
from plantdiseasedetection.utils.main_utils import read_yaml_file
from plantdiseasedetection.exception import AppException
from plantdiseasedetection.entity.config_entity import ModelTrainerConfig
from plantdiseasedetection.entity.artifacts_entity import ModelTrainerArtifact



class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
    ):
        self.model_trainer_config = model_trainer_config


    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")

        try:
            logging.info("Unzipping Data")
            with zipfile.ZipFile('data.zip', 'r') as zip_ref:
                zip_ref.extractall()
            os.remove("data.zip")
            
            with open("data.yaml", "r") as stream:
                num_classes = str(yaml.safe_load(stream)["nc"])

            model_config_file_name = self.model_trainer_config_file_name.weight_name.split(",")[0]
   
            print(model_config_file_name)

            config = read_yaml_file(f"yolov9/models/detect/{model_config_file_name}.yaml")

            config['nc'] = int(num_classes)

            with open(f'yolov9/models/detect/custom_{model_config_file_name}.yaml', 'w') as f:
                yaml.dump(config, f)
                
            config['nc'] = int(num_classes)

            os.system(f"cd yolov9/ && python train.py --workers 8  --batch {self.model_trainer_config.batch_size} --img 640  --epochs {self.model_trainer_config.no_epochs} --data../data.yaml --weights {self.model_trainer_config.weight_name} --device 0 --cfg.yolov9/models/detect/gelan-c.yaml  --hyp yolov9/data/hyps/hyp.scratch-high.yaml")
            os.system("cp yolov9/runs/train/exp/weights/best.pt yolov9/")
            os.makedirs(self.model_trainer_config.model_trainer_dir, exist_ok=True)
            os.system(f"cp yolov9/runs/train/exp/weights/best.pt {self.model_trainer_config.model_trainer_dir}/")


            os.system("rm -rf yolov9/runs")
            os.system("rm -rf train")
            os.system("rm -rf valid")
            os.system("rm -rf test")
            os.system("rm -rf yolov9/data.yaml")

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path="yolov9/best.pt"
            )

        except Exception as e:
            raise AppException(e, sys)



