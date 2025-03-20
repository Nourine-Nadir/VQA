import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.image_list import ImageList

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class BiLSTMForRegression(torch.nn.Module):
    def __init__(self,lr, *args, **kwargs):
        super(BiLSTMForRegression, self).__init__(*args, **kwargs)


        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=1024,
            num_layers=8,  # LSTM layers
            bidirectional=True,  # Bidirectional LSTM
            dropout=0.3,  # Dropout for LSTM
            batch_first=True  # Input shape: (batch, seq_len, features)
        )

        self.fc = nn.Sequential(

            nn.Linear(2048, 1024),  # Bidirectional LSTM output size is 1024 * 2
            nn.ReLU(),  # ReLU activation
            nn.Dropout(0.3),  # Dropout
            nn.LayerNorm(1024), # Layer Norm for stability
            nn.Linear(1024, 512),
            nn.ReLU(),  # ReLU activation
            nn.Dropout(0.3),  # Dropout
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.ReLU(),  # ReLU activation
            nn.Dropout(0.3),  # Dropout
            nn.LayerNorm(256),
            nn.Linear(256, 1),
            # Output a single regression value

            nn.Identity()  # Linear activation (no change)
        )
        self.norm = nn.LayerNorm(2048)

        # Loss function and optimizer
        self.mse_loss_fn = nn.MSELoss()  # Mean Squared Error loss for regression
        self.ranking_loss_fn = RankingLoss()  # Mean Squared Error loss for regression
        self.optimizer = optim.Adam(self.parameters(),lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                              'min',
                                                              patience=5,
                                                              )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.apply(weights_init_)

    def forward(self, x):
        batch_size, nb_frames, emb_dim, _, _ = x.shape
        x = x.squeeze()

        x = self.norm(x)
        # LSTM forward pass
        x, _ = self.lstm(x)

        x = x[:, -1, :]  # Take the output of the last time step

        # Fully connected layers
        x = self.fc(x)  # Output shape: (batch_size, 1)
        return x

    def get_lr(self):
        return self.scheduler.get_last_lr()[0]

    def lr_decay(self, ep_loss):
        self.scheduler.step(ep_loss)

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        torch.save(self.state_dict(), path)

    def load_model(self, path, map_location=None):
        if map_location is None:
            map_location = self.device  # Use the model's current device
        self.load_state_dict(torch.load(path, map_location=map_location, weights_only=True))

class RankingLoss(nn.Module):
    def __init__(self, margin=0.1):
        super().__init__()
        self.margin = margin
    def forward(self, predicted_a, predicted_b, target_a, target_b):

        diff = (predicted_a - predicted_b) * (target_a - target_b)
        loss = torch.clamp(self.margin - diff, min=0)

        return loss.mean()

class RCnn(torch.nn.Module):
    def __init__(self, patch_size = 224,*args):
        super(RCnn, self).__init__(*args)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.patch_size = patch_size

        self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = torch.nn.Sequential(*list(self.backbone.children())[:-2])
        self.backbone.out_channels = 2048

        self.rpn_anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5
        )
        self.roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=7,  # Output size of the RoI pooling
            sampling_ratio=2
        )

        # Create the Faster R-CNN model
        self.rcnn_model = FasterRCNN(
            self.backbone,
            num_classes=91,  #  (not used for feature extraction)
            rpn_anchor_generator=self.rpn_anchor_generator,
            box_roi_pool=self.roi_pooler
        )
        self.rcnn_model.eval()
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.to(self.device)

    def extract_rcnn_features(self, frame):

        with torch.no_grad():
            # Pass the frame through the R-CNN backbone
            features = self.rcnn_model.backbone(frame.unsqueeze(0))  # Add batch dimension
            # Generate region proposals
            features = {'0': features}  # Wrap in a dictionary with key '0'
            image_list = ImageList(frame.unsqueeze(0), [(frame.shape[1], frame.shape[2])])

            proposals, _ = self.rcnn_model.rpn(image_list, features)
            # Extract RoI features
            roi_features = self.rcnn_model.roi_heads.box_roi_pool(features, proposals, [frame.shape[1:]])

            roi_features = self.global_avg_pool(roi_features)

            # Average features across regions
            roi_features = roi_features.mean(dim=0)  # Shape: (2048,)

        return roi_features

    def forward(self, x):

        batch_size, channels, num_frames, height, width = x.shape

        # Reshape input to (batch_size * num_frames, channels, height, width)
        x = x.view(-1, channels, height, width)

        # Extract features for each frame using R-CNN
        features = []
        for frame in x:
            frame_features = self.extract_rcnn_features(frame)  # Shape: (2048,)
            features.append(frame_features)
        features = torch.stack(features)  # Shape: (batch_size * num_frames, 2048)

        # Reshape features back to (batch_size, num_frames, 2048, 1, 1)
        features = features.view(batch_size, num_frames, 2048, 1, 1)

        return features

