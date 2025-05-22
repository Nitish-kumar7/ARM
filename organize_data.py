import os
import shutil
import pandas as pd

def create_folders():
    """Create necessary folders for data organization"""
    # Create main folders
    os.makedirs('data/Ultrasound_images/normal', exist_ok=True)
    os.makedirs('data/Ultrasound_images/arm', exist_ok=True)
    
    print("Created folders:")
    print("- data/Ultrasound_images/normal")
    print("- data/Ultrasound_images/arm")

def move_images():
    """Move images to their respective folders"""
    # Load the CSV file
    df = pd.read_csv('data/Routine_anal_pit.csv')
    
    # Get list of all images
    all_images = os.listdir('data/Ultrasound_images')
    
    print("\nPlease categorize your images:")
    print("1. Move normal ultrasound images to 'data/Ultrasound_images/normal'")
    print("2. Move ARM ultrasound images to 'data/Ultrasound_images/arm'")
    print("\nYou can do this manually by:")
    print("1. Opening the 'data/Ultrasound_images' folder")
    print("2. Creating two folders: 'normal' and 'arm'")
    print("3. Moving each image to the appropriate folder based on whether it shows ARM or not")
    
    print(f"\nTotal images found: {len(all_images)}")

if __name__ == "__main__":
    print("=== Data Organization Helper ===")
    create_folders()
    move_images()
    print("\nAfter organizing the images, run 'python train.py' to start training the model.") 