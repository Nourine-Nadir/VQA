import os
import torch
from orca.orca_platform import datadir
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from load_data import load_video_frames
import re
class VideoQualityDataset(Dataset):
    def __init__(self, data_path, labels_path=None, num_frames=25, resize_size=224):
        """
        Args:
            data_path (str): Path to the directory containing video files.
            labels_path (str): Path to the CSV file containing video labels (flickr_id and mos).
            num_frames (int): Number of frames to sample from each video.
            resize_size (tuple): Size to resize video frames (height, width).
        """
        self.data_path = data_path
        self.labels_path = labels_path
        self.num_frames = num_frames
        self.resize_size = resize_size

        # Load labels and convert 'flickr_id' to strings
        if not (labels_path == None):

            self.labels = pd.read_csv(labels_path)
            self.labels['filename'] = self.labels['filename'].astype(str)  # Convert to strings
            self.labels.set_index('filename', inplace=True)  # Set 'flickr_id' as the index

        # Get list of video files
        self.video_files = [f for f in os.listdir(data_path) if f.endswith('.mp4')]



    def __len__(self):
        return len(self.video_files)


    def __getitem__(self, idx):
        # Get video filename
        vid = self.video_files[idx]

        # Extract flickr_id from the filename
        filename = vid.split('.')[0]  # Remove the file extension

        if not (self.labels_path == None):
            # Get the MOS for the video
            if filename in self.labels.index:
                mos = self.labels.loc[filename]  # Access MOS directly from the Series
            else:
                raise ValueError(f"filename {filename} not found in labels.")

        # Load video frames
        path_file = os.path.join(self.data_path, vid)
        video_tensor = load_video_frames(path_file, num_frames=self.num_frames, img_size=self.resize_size)  # (T, H, W, C)
        if (self.labels_path == None):
            return  video_tensor, 0

        else:
            return video_tensor, torch.tensor(mos, dtype=torch.float32)

class FeatureDataset(Dataset):
    def __init__(self, data_dir):
        """
        Args:
            features_dir (str): Directory containing feature batch files.
            labels_dir (str): Directory containing label batch files.
        """

        # List all feature and label batch files
        self.feature_files = [f for f in os.listdir(data_dir) if f.startswith("features_batch_")]
        self.label_files = [f for f in os.listdir(data_dir) if f.startswith("labels_batch_")]

        # Sort files numerically based on the batch index
        self.feature_files = sorted(self.feature_files, key=self.extract_batch_index)
        self.label_files = sorted(self.label_files, key=self.extract_batch_index)

        # Construct full paths
        self.feature_files = [os.path.join(data_dir, f) for f in self.feature_files]
        self.label_files = [os.path.join(data_dir, f) for f in self.label_files]

        print(f'features files {self.feature_files}')
        # Load all features and labels into memory
        self.features = []
        self.labels = []
        for feat_file, label_file in zip(self.feature_files, self.label_files):
            self.features.extend(torch.load(feat_file, weights_only=False))
            self.labels.extend(torch.load(label_file, weights_only=False))


    def extract_batch_index(self, filename):
        """
        Extract the numerical batch index from a filename.
        Example: "features_batch_100.pt" -> 100
        """
        match = re.search(r'features_batch_(\d+)\.pt', filename) or re.search(r'labels_batch_(\d+)\.pt', filename)
        if match:
            return int(match.group(1))
        return -1  # Fallback for invalid filenames

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]  # Example usage
