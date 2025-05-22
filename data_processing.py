import os
import pandas as pd
import shutil
from PIL import Image
import torch
from torchvision import transforms
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self, data_dir='data', output_dir='processed_data'):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.csv_path = os.path.join(data_dir, 'Routine_anal_pit.csv')
        self.images_dir = os.path.join(data_dir, 'Ultrasound_images')
        
        # Create output directories
        self.train_dir = os.path.join(output_dir, 'train')
        self.val_dir = os.path.join(output_dir, 'val')
        self.test_dir = os.path.join(output_dir, 'test')
        
        # Image preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    
    def create_directory_structure(self):
        """Create the necessary directory structure for processed data"""
        for dir_path in [self.train_dir, self.val_dir, self.test_dir]:
            for class_name in ['normal', 'arm']:
                os.makedirs(os.path.join(dir_path, class_name), exist_ok=True)
    
    def load_metadata(self):
        """Load and process the CSV metadata file"""
        df = pd.read_csv(self.csv_path)
        return df
    
    def preprocess_image(self, image_path):
        """Preprocess a single image"""
        try:
            image = Image.open(image_path).convert('RGB')
            return self.transform(image)
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return None
    
    def organize_data(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """Organize data into train/val/test splits"""
        # Create directory structure
        self.create_directory_structure()
        
        # Load metadata
        df = self.load_metadata()
        
        # TODO: Add logic to determine normal vs ARM cases
        # For now, we'll need to manually label the data or get labels from another source
        
        print("Data organization completed. Please ensure proper labeling of normal vs ARM cases.")
        print(f"Total images found: {len(df)}")
        
    def get_data_loaders(self, batch_size=32):
        """Create PyTorch data loaders for training"""
        # TODO: Implement data loaders once we have labeled data
        pass

if __name__ == "__main__":
    processor = DataProcessor()
    processor.organize_data() 