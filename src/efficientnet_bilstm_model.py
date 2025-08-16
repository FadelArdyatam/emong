import torch
import torch.nn as nn
import torchvision.models as models

class EfficientNetBiLSTM(nn.Module):
    def __init__(self, num_emotion_classes, efficientnet_version='b0', lstm_hidden_size=256, lstm_num_layers=2, dropout_prob=0.5):
        super(EfficientNetBiLSTM, self).__init__()

        # 1. EfficientNet Backbone for Feature Extraction
        if efficientnet_version == 'b0':
            self.efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            # Remove the classifier head to use it as a feature extractor
            self.efficientnet.classifier = nn.Identity()
            feature_dim = 1280 # Output features for EfficientNetB0
        elif efficientnet_version == 'b1':
            self.efficientnet = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1)
            self.efficientnet.classifier = nn.Identity()
            feature_dim = 1280 # Output features for EfficientNetB1
        # Add more versions as needed
        else:
            raise ValueError(f"EfficientNet version {efficientnet_version} not supported.")

        # 2. BiLSTM Head for Temporal Modeling
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True, # Input shape: (batch_size, sequence_length, input_size)
            bidirectional=True
        )

        # 3. Classification Head
        # Bi-directional LSTM outputs hidden_size * 2 features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(lstm_hidden_size * 2, num_emotion_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, sequence_length, channels, height, width)
        batch_size, seq_len, C, H, W = x.size()

        # Reshape for EfficientNet: (batch_size * seq_len, channels, height, width)
        # Process each frame through EfficientNet
        x_reshaped = x.view(batch_size * seq_len, C, H, W)
        features = self.efficientnet(x_reshaped) # (batch_size * seq_len, feature_dim)

        # Reshape features for LSTM: (batch_size, sequence_length, feature_dim)
        features = features.view(batch_size, seq_len, -1)

        # Pass through LSTM
        lstm_out, _ = self.lstm(features) # lstm_out shape: (batch_size, sequence_length, lstm_hidden_size * 2)

        # Take the output from the last time step for classification
        # Or, you could average/max pool across the sequence dimension
        last_time_step_output = lstm_out[:, -1, :] # (batch_size, lstm_hidden_size * 2)

        # Classify
        output = self.classifier(last_time_step_output)
        return output

if __name__ == '__main__':
    # Example Usage:
    # Assuming input sequence of 5 frames, each 3x224x224 (RGB image)
    dummy_input = torch.randn(2, 5, 3, 224, 224) # Batch size 2, sequence length 5

    num_classes = 7 # Example: Happy, Sad, Angry, Neutral, etc.
    model = EfficientNetBiLSTM(num_emotion_classes=num_classes)

    output = model(dummy_input)
    print(f"Output shape: {output.shape}") # Expected: (batch_size, num_emotion_classes)
    # Expected output shape: torch.Size([2, 7])
