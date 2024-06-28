from datasets import load_dataset, DatasetDict, Dataset
import pandas as pd
import os

class ControlNetDataset():
    def __init__(self, huggingface_dataset_path):
        self.huggingface_dataset_path = huggingface_dataset_path

    def convert_image_mode(self, image):
        if image.mode != 'RGB':
            image = image.convert("RGB")
        return image
    
    def get_dataset(self):
        dataset = load_dataset(self.huggingface_dataset_path)
        return dataset
    
    def process_dataset(self):
        dataset = self.get_dataset()
        data_dict = {
            'images':dataset['train']['images'],
            'condition_images':dataset['train']['conditions'],
            'prompt':dataset['train']['prompt']
            }
        data_dict_pd = pd.DataFrame(data_dict)
        images = ['images', 'condition_images']
        for img_key in images:
            data_dict_pd[img_key] = data_dict_pd[img_key].apply(self.convert_image_mode)
        dataset_dict_final = Dataset.from_dict(data_dict_pd)
        dataset_dict_final = DatasetDict({"train": dataset_dict_final})
        return dataset_dict_final
    
    def save_dataset(self, dir_path_to_save_dataset, dataset_dict_final):
        if not os.path.exists(dir_path_to_save_dataset):
            os.makedirs(dir_path_to_save_dataset, exist_ok=True)
        dataset_dict_final.save_to_disk(dir_path_to_save_dataset)
        print(f"Dataset is saved successfully at {dir_path_to_save_dataset}")
                