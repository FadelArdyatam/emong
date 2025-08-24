#!/usr/bin/env python3
"""
Training script untuk YOLO Emotion Detection Model
Menggunakan FER2013 dataset dengan YOLO format
"""
import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import yaml
from pathlib import Path

def create_yolo_dataset_structure():
    """
    Create YOLO dataset structure for emotion detection
    """
    print("📁 Creating YOLO dataset structure...")
    
    # Dataset structure
    dataset_path = "yolo_emotion_dataset"
    os.makedirs(dataset_path, exist_ok=True)
    
    # Create train/val/test splits
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(dataset_path, split)
        os.makedirs(split_path, exist_ok=True)
        os.makedirs(os.path.join(split_path, 'images'), exist_ok=True)
        os.makedirs(os.path.join(split_path, 'labels'), exist_ok=True)
    
    # Create dataset.yaml
    dataset_config = {
        'path': os.path.abspath(dataset_path),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': 7,  # Number of classes
        'names': ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprised']
    }
    
    with open(os.path.join(dataset_path, 'dataset.yaml'), 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"✅ Dataset structure created at {dataset_path}")
    return dataset_path

def convert_fer2013_to_yolo(fer2013_csv_path, output_path):
    """
    Convert FER2013 dataset to YOLO format
    """
    print("🔄 Converting FER2013 to YOLO format...")
    
    if not os.path.exists(fer2013_csv_path):
        print(f"❌ FER2013 CSV not found at {fer2013_csv_path}")
        print("📥 Download FER2013 from: https://www.kaggle.com/datasets/msambare/fer2013")
        return False
    
    # Emotion mapping
    emotion_map = {
        0: 'Angry',
        1: 'Disgust', 
        2: 'Fear',
        3: 'Happy',
        4: 'Sad',
        5: 'Surprised',
        6: 'Neutral'
    }
    
    # Read FER2013 CSV
    import pandas as pd
    df = pd.read_csv(fer2013_csv_path)
    
    # Split data (80% train, 10% val, 10% test)
    train_size = int(0.8 * len(df))
    val_size = int(0.1 * len(df))
    
    train_df = df[:train_size]
    val_df = df[train_size:train_size + val_size]
    test_df = df[train_size + val_size:]
    
    splits = {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }
    
    # Process each split
    for split_name, split_df in splits.items():
        print(f"📊 Processing {split_name} split: {len(split_df)} images")
        
        split_images_path = os.path.join(output_path, split_name, 'images')
        split_labels_path = os.path.join(output_path, split_name, 'labels')
        
        for idx, row in split_df.iterrows():
            try:
                # Get emotion and pixels
                emotion = row['emotion']
                pixels = row['pixels']
                
                # Convert pixels to image
                pixels = [int(p) for p in pixels.split()]
                img_array = np.array(pixels, dtype=np.uint8).reshape(48, 48)
                
                # Resize to 640x640 (YOLO standard)
                img_resized = cv2.resize(img_array, (640, 640))
                
                # Convert to RGB
                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
                
                # Save image
                img_filename = f"emotion_{idx:06d}.jpg"
                img_path = os.path.join(split_images_path, img_filename)
                cv2.imwrite(img_path, img_rgb)
                
                # Create YOLO label
                # Since FER2013 is centered face, we'll create a bounding box
                # that covers most of the image (face region)
                img_width, img_height = 640, 640
                
                # Face bounding box (center coordinates, width, height in YOLO format)
                # Assuming face takes up 80% of the image
                face_size = 0.8
                x_center = 0.5  # Center of image
                y_center = 0.5  # Center of image
                width = face_size
                height = face_size
                
                # YOLO format: class x_center y_center width height
                label_line = f"{emotion} {x_center} {y_center} {width} {height}\n"
                
                # Save label
                label_filename = f"emotion_{idx:06d}.txt"
                label_path = os.path.join(split_labels_path, label_filename)
                
                with open(label_path, 'w') as f:
                    f.write(label_line)
                
            except Exception as e:
                print(f"❌ Error processing image {idx}: {e}")
                continue
    
    print("✅ FER2013 conversion completed!")
    return True

def train_yolo_emotion_model(dataset_path, model_size='s'):
    """
    Train YOLO model for emotion detection
    """
    print(f"🚀 Training YOLO{model_size} emotion detection model...")
    
    # Initialize YOLO model
    model = YOLO(f'yolov8{model_size}.pt')  # Start with pretrained YOLOv8
    
    # Training configuration
    training_args = {
        'data': os.path.join(dataset_path, 'dataset.yaml'),
        'epochs': 100,
        'imgsz': 640,
        'batch': 16,
        'device': '0' if torch.cuda.is_available() else 'cpu',
        'workers': 4,
        'patience': 20,
        'save': True,
        'save_period': 10,
        'cache': True,
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'weight_decay': 0.0005,
        'momentum': 0.937,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,  # Box loss gain
        'cls': 0.5,  # Class loss gain
        'dfl': 1.5,  # DFL loss gain
        'pose': 12.0,  # Pose loss gain
        'kobj': 1.0,  # Keypoint obj loss gain
        'label_smoothing': 0.0,
        'nbs': 64,  # Nominal batch size
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.0,
        'val': True,
        'plots': True
    }
    
    # Start training
    try:
        results = model.train(**training_args)
        print("✅ Training completed successfully!")
        
        # Save the trained model
        model_path = f"models/yolo_emotion_detection_v8{model_size}.pt"
        os.makedirs("models", exist_ok=True)
        model.save(model_path)
        print(f"💾 Model saved to {model_path}")
        
        return model_path
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None

def create_custom_yolo_config():
    """
    Create custom YOLO configuration for emotion detection
    """
    print("⚙️ Creating custom YOLO configuration...")
    
    config = {
        'nc': 7,  # Number of classes
        'depth_multiple': 1.0,  # Model depth multiple
        'width_multiple': 1.0,  # Model width multiple
        'anchors': [
            [10, 13, 16, 30, 33, 23],  # P3/8
            [30, 61, 62, 45, 59, 119],  # P4/16
            [116, 90, 156, 198, 373, 326]  # P5/32
        ],
        'backbone': [
            # [from, number, module, args]
            [-1, 1, 'Conv', [64, 3, 2]],  # 0-P1/2
            [-1, 1, 'Conv', [128, 3, 2]],  # 1-P2/4
            [-1, 3, 'C2', [128]],
            [-1, 1, 'Conv', [256, 3, 2]],  # 3-P3/8
            [-1, 6, 'C2', [256]],
            [-1, 1, 'Conv', [512, 3, 2]],  # 5-P4/16
            [-1, 9, 'C2', [512]],
            [-1, 1, 'Conv', [1024, 3, 2]],  # 7-P5/32
            [-1, 3, 'C2', [1024]],
            [-1, 1, 'SPPF', [1024, 5]],  # 9
        ],
        'head': [
            [-1, 1, 'Conv', [512, 1, 1]],
            [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
            [[-1, 6], 1, 'Concat', [1]],  # cat backbone P4
            [-1, 3, 'C2', [512, False]],  # 13
            
            [-1, 1, 'Conv', [256, 1, 1]],
            [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
            [[-1, 4], 1, 'Concat', [1]],  # cat backbone P3
            [-1, 3, 'C2', [256, False]],  # 17 (P3/8)
            
            [-1, 1, 'Conv', [256, 3, 2]],
            [[-1, 14], 1, 'Concat', [1]],  # cat head P4
            [-1, 3, 'C2', [512, False]],  # 20 (P4/16)
            
            [-1, 1, 'Conv', [512, 3, 2]],
            [[-1, 10], 1, 'Concat', [1]],  # cat head P5
            [-1, 3, 'C2', [1024, False]],  # 23 (P5/32)
            
            [[17, 20, 23], 1, 'Detect', [7, 'anchors']],  # Detect(P3, P4, P5)
        ]
    }
    
    # Save configuration
    config_path = "yolo_emotion_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✅ Custom configuration saved to {config_path}")
    return config_path

def main():
    """Main training pipeline"""
    print("🎯 YOLO EMOTION DETECTION TRAINING PIPELINE")
    print("=" * 60)
    
    # Step 1: Create dataset structure
    dataset_path = create_yolo_dataset_structure()
    
    # Step 2: Convert FER2013 to YOLO format
    fer2013_path = "fer2013.csv"  # Update this path
    if os.path.exists(fer2013_path):
        convert_fer2013_to_yolo(fer2013_path, dataset_path)
    else:
        print(f"⚠️ FER2013 dataset not found at {fer2013_path}")
        print("📥 Please download FER2013 dataset first")
        print("🔗 Download from: https://www.kaggle.com/datasets/msambare/fer2013")
        return
    
    # Step 3: Create custom configuration
    config_path = create_custom_yolo_config()
    
    # Step 4: Train model
    model_path = train_yolo_emotion_model(dataset_path, model_size='s')
    
    if model_path:
        print("\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print(f"📁 Model saved to: {model_path}")
        print(f"📊 Dataset location: {dataset_path}")
        print(f"⚙️ Config file: {config_path}")
        
        print("\n🚀 NEXT STEPS:")
        print("1. Test the trained model with yolo_emotion_detector.py")
        print("2. Integrate with your existing EMONG system")
        print("3. Fine-tune hyperparameters if needed")
        print("4. Test real-time performance")
    else:
        print("\n❌ Training failed. Please check the error messages above.")

if __name__ == "__main__":
    main() 