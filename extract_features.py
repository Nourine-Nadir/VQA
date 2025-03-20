import torch.cuda as cuda
import pandas as pd
from models import *
from VQA_datasets import VideoQualityDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def extract_and_store_features(config, res_model, dataset, output_dir):
    res_model.eval()  # Set ResNet to evaluation mode
    os.makedirs(output_dir, exist_ok=True)  # Create output directory

    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    start_event = cuda.Event(enable_timing=True)
    end_event = cuda.Event(enable_timing=True)

    runtimes = []
    for batch_idx, (videos, mos) in enumerate(tqdm(dataloader, desc="Extracting features")):
        videos = videos.to(res_model.device)  # Move input tensor to the same device as the model
        start_event.record()

        with torch.no_grad():
            features = res_model(videos)  # Extract features: (batch_size, num_frames, 2048)
            # End timer
        end_event.record()

        # Synchronize to ensure accurate timing
        cuda.synchronize()
        # Calculate elapsed time in milliseconds
        elapsed_time = start_event.elapsed_time(end_event) / 1000  # Convert to seconds
        runtimes.append(elapsed_time)
        # Calculate average runtime
        average_runtime = sum(runtimes) / len(runtimes)
        print(f"Average runtime per video: {average_runtime/config.batch_size:.4f} seconds")
        print(f"Total runtime for {len(runtimes)} batches: {sum(runtimes):.4f} seconds")
        # Save features and labels for this batch
        batch_features_path = os.path.join(output_dir, f"features_batch_{batch_idx}.pt")
        batch_labels_path = os.path.join(output_dir, f"labels_batch_{batch_idx}.pt")
        torch.save(features.cpu(), batch_features_path)
        torch.save(mos.cpu(), batch_labels_path)

    print(f"Features saved to {output_dir}")


# Example usage
def run_feature_extraction(config):
    res_model = RCnn().to(DEVICE)  # Move ResNet to the appropriate device
    train_labels = pd.read_csv(config.train_labels_path)

    # Ensure filename column is string type
    train_labels['filename'] = train_labels['filename'].astype(str)

    # Remove "train/" prefix
    train_labels['filename'] = train_labels['filename']

    # Set the cleaned filename as the index
    train_labels.set_index('filename', inplace=True)

    # Assuming you have a list of video filenames and corresponding MOS labels
    train_files = sorted([f for f in os.listdir(config.train_data_path) if f.endswith('.mp4')][:2])
    train_mos = [train_labels.loc[str('train/' + f), 'score'] for f in train_files][:2]
    val_labels = pd.read_csv(config.val_labels_path)

    # Ensure filename column is string type
    val_labels['filename'] = val_labels['filename'].astype(str)

    # Remove "train/" prefix
    val_labels['filename'] = val_labels['filename']

    # Set the cleaned filename as the index
    val_labels.set_index('filename', inplace=True)

    # Assuming you have a list of video filenames and corresponding MOS labels
    val_files = sorted([f for f in os.listdir(config.val_data_path) if f.endswith('.mp4')][:2])
    val_mos = [val_labels.loc[str('val/' + f), 'score'] for f in val_files][:2]

    test_files = sorted([f for f in os.listdir(config.test_data_path) if f.endswith('.mp4')][:2])

    # Create training dataset
    test_dataset = VideoQualityDataset(
        data_path=config.test_data_path,
        num_frames=config.num_frames,  # Adjust as needed
        resize_size=config.img_size)

    # Create training dataset
    train_dataset = VideoQualityDataset(
        data_path=config.train_data_path,
        labels_path=config.train_labels_path,
        num_frames=config.num_frames,  # Adjust as needed
        resize_size=config.img_size)

    train_dataset.video_files = train_files  # Override with training files
    train_dataset.labels = pd.Series(train_mos, index=[f.split('.')[0] for f in train_files])

    # Create validation dataset
    val_dataset = VideoQualityDataset(
        data_path=config.val_data_path,
        labels_path=config.val_labels_path,
        num_frames=config.num_frames,  # Adjust as needed
        resize_size=config.img_size)

    val_dataset.video_files = val_files  # Override with validation files
    val_dataset.labels = pd.Series(val_mos, index=[f.split('.')[0] for f in val_files])

    # Extract and store features for training and validation sets
    # extract_and_store_features(config, res_model, train_dataset,
    #                            output_dir=config.train_data_dir)
    # extract_and_store_features(config, res_model, val_dataset,
    #                            output_dir=config.val_data_dir)
    extract_and_store_features(config, res_model, test_dataset,
                               output_dir=config.test_data_dir)
