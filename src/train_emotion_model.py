import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from efficientnet_bilstm_model import EfficientNetBiLSTM # Import your model

# --- Configuration ---
# Define your dataset path here.
# Example: If your dataset is structured as data/train/happy/img.jpg
DATA_DIR = 'data' # This should be relative to where you run the script, or an absolute path
MODEL_SAVE_PATH = 'models/emotion_efficientnet_bilstm_trained.pth' # Path to save the trained model
LOG_FILE = 'training_log.txt'

# NUM_EMOTION_CLASSES will be determined by the dataset
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50 # You might need to adjust this
IMAGE_SIZE = (224, 224) # EfficientNet expects 224x224 for b0/b1
EFFICIENTNET_VERSION = 'b0' # 'b0' or 'b1'
LSTM_HIDDEN_SIZE = 256
LSTM_NUM_LAYERS = 2
DROPOUT_PROB = 0.5

# --- 1. Custom Dataset Class ---
class EmotionDataset(Dataset):
    def __init__(self, data_dir, transform=None, test_size=0.2, random_state=42, subset='train', label_encoder=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Collect all image paths and labels
        all_image_paths = []
        all_labels_str = [] # Store string labels for encoding
        for emotion_label in os.listdir(data_dir):
            emotion_path = os.path.join(data_dir, emotion_label)
            if os.path.isdir(emotion_path):
                for img_name in os.listdir(emotion_path):
                    img_path = os.path.join(emotion_path, img_name)
                    if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                        all_image_paths.append(img_path)
                        all_labels_str.append(emotion_label)

        # Use provided label_encoder or create a new one
        if label_encoder:
            self.label_encoder = label_encoder
        else:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(all_labels_str) # Fit on all unique string labels

        encoded_labels = self.label_encoder.transform(all_labels_str)

        # Split data into training and validation sets
        # Ensure consistent split across runs and subsets
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            all_image_paths, encoded_labels, test_size=test_size, random_state=random_state, stratify=encoded_labels
        )

        if subset == 'train':
            self.image_paths = train_paths
            self.labels = train_labels
        elif subset == 'val':
            self.image_paths = val_paths
            self.labels = val_labels
        else:
            raise ValueError("Subset must be 'train' or 'val'")

        print(f"Loaded {len(self.image_paths)} images for {subset} subset.")
        print(f"Emotion classes: {self.label_encoder.classes_}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Read image using OpenCV and convert to RGB
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found or corrupted: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)

        # For sequence models, we typically need a sequence of frames.
        # For now, we'll treat each image as a single-frame sequence.
        # The model expects (batch_size, sequence_length, channels, height, width)
        # So, we add a sequence_length dimension of 1.
        img = img.unsqueeze(0) # Add sequence_length dimension

        return img, torch.tensor(label, dtype=torch.long)

# --- 2. Data Transformations (same as before) ---
train_transforms = transforms.Compose([
    transforms.ToPILImage(), # Convert numpy array to PIL Image for torchvision transforms
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(), # Converts to [0, 1] and C, H, W
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet normalization
])

val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- 3. Setup Device (same as before) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 4. Training and Validation Functions (same as before) ---
def train_epoch(model, dataloader, criterion, optimizer, device, epoch, log_file):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

        if (batch_idx + 1) % 10 == 0: # Log every 10 batches
            log_message = (f"Epoch [{epoch}/{NUM_EPOCHS}], Batch [{batch_idx+1}/{len(dataloader)}], "
                           f"Loss: {loss.item():.4f}")
            print(log_message)
            with open(log_file, 'a') as f:
                f.write(log_message + '\n')

    epoch_loss = running_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples
    log_message = (f"Epoch {epoch} Training Loss: {epoch_loss:.4f}, "
                   f"Training Accuracy: {epoch_accuracy:.4f}")
    print(log_message)
    with open(log_file, 'a') as f:
        f.write(log_message + '\n')
    return epoch_loss, epoch_accuracy

def validate_epoch(model, dataloader, criterion, device, epoch, log_file):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    epoch_loss = running_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples
    log_message = (f"Epoch {epoch} Validation Loss: {epoch_loss:.4f}, "
                   f"Validation Accuracy: {epoch_accuracy:.4f}")
    print(log_message)
    with open(log_file, 'a') as f:
        f.write(log_message + '\n')

    # Optional: Print classification report
    report = classification_report(all_labels, all_predictions, target_names=dataloader.dataset.label_encoder.classes_, zero_division=0)
    log_message = f"Classification Report for Epoch {epoch}:\n{report}"
    print(log_message)
    with open(log_file, 'a') as f:
        f.write(log_message + '\n')

    return epoch_loss, epoch_accuracy

# --- 5. Main Training Loop ---
def main():
    # Create datasets and dataloaders
    print("Loading datasets...")
    # First, create a dummy dataset to fit the LabelEncoder and get NUM_EMOTION_CLASSES
    # This is a workaround to get the number of classes before initializing the model
    # A better approach would be to pass the class names directly or use a config file
    temp_dataset = EmotionDataset(data_dir=DATA_DIR, subset='train')
    global NUM_EMOTION_CLASSES
    NUM_EMOTION_CLASSES = len(temp_dataset.label_encoder.classes_)
    print(f"Detected {NUM_EMOTION_CLASSES} emotion classes.")

    train_dataset = EmotionDataset(data_dir=DATA_DIR, transform=train_transforms, subset='train',
                                   label_encoder=temp_dataset.label_encoder)
    val_dataset = EmotionDataset(data_dir=DATA_DIR, transform=val_transforms, subset='val',
                                 label_encoder=temp_dataset.label_encoder)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Initialize model
    print("Initializing model...")
    model = EfficientNetBiLSTM(
        num_emotion_classes=NUM_EMOTION_CLASSES,
        efficientnet_version=EFFICIENTNET_VERSION,
        lstm_hidden_size=LSTM_HIDDEN_SIZE,
        lstm_num_layers=LSTM_NUM_LAYERS,
        dropout_prob=DROPOUT_PROB
    ).to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Create model save directory if it doesn't exist
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    best_val_accuracy = 0.0

    # Clear previous log file
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    print("Starting training...")
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer, device, epoch, LOG_FILE)
        val_loss, val_accuracy = validate_epoch(model, val_loader, criterion, device, epoch, LOG_FILE)

        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            log_message = f"Model saved! Validation Accuracy improved to {best_val_accuracy:.4f}"
            print(log_message)
            with open(LOG_FILE, 'a') as f:
                f.write(log_message + '\n')

    print("Training finished!")
    print(f"Best Validation Accuracy: {best_val_accuracy:.4f}")
    print(f"Trained model saved to: {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    main()
