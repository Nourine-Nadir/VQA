from utils import plot_metrics, calculate_metrics
import torch.cuda as cuda
import pandas as pd
from models import *
from VQA_datasets import FeatureDataset
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




def test(config):

    # Load feature datasets
    test_dataset = FeatureDataset(data_dir=config.test_data_dir)
    # Create DataLoader
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    # Initialize BiSLTM model
    model = BiLSTMForRegression(

        lr=config.learning_rate,

    ).to(DEVICE)
    model.load_model(path='saved_models/best_BiLSTM')
    metrics = {
        'SROCC': [],
        'PLCC': [],
        'KROCC': [],
        'RMSE': [],
    }

    # Test
    runtimes = []

    # Create CUDA events for timing
    start_event = cuda.Event(enable_timing=True)
    end_event = cuda.Event(enable_timing=True)
    model.eval()
    val_preds, val_labels = [], []

    test_data_example_file = config.test_example_csv
    with torch.no_grad():
        running_loss = 0.0
        y_pred = []
        for features, mos in tqdm(test_loader, desc="Test"):
            features = features.to(DEVICE)
            mos = mos.to(DEVICE)
            # Start timer
            start_event.record()

            outputs = model(features)

            # End timer
            end_event.record()

            # Synchronize to ensure accurate timing
            cuda.synchronize()

            # Calculate elapsed time in milliseconds
            elapsed_time = start_event.elapsed_time(end_event) / 1000  # Convert to seconds
            runtimes.append(elapsed_time)

            val_loss = model.mse_loss_fn(outputs.squeeze(), mos)

            y_pred.extend(outputs.cpu().numpy())

            val_preds.extend(outputs.cpu().numpy())
            val_labels.extend(mos.cpu().numpy())
            running_loss += val_loss * features.size(0)

        # Calculate average runtime
        average_runtime = sum(runtimes) / len(runtimes)
        print(f"Average runtime per video: {average_runtime:.4f} seconds")
        print(f"Total runtime for {len(runtimes)} videos: {sum(runtimes):.4f} seconds")

        outputs_df = pd.read_csv(test_data_example_file)
        outputs_df['score'] = np.squeeze(np.array(y_pred))
        outputs_df.set_index('filename', inplace=True)
        outputs_df.to_csv('prediction.csv')
        # epoch_loss = running_loss / len(test_dataset)
        #
        # print(f'epoch loss {epoch_loss}')
        # model.lr_decay(epoch_loss)
        # print(f'LR : {model.get_lr()}')
        #
        # # Calculate metrics
        # srocc, plcc, krocc, rmse = calculate_metrics(val_labels, val_preds)
        # metrics['SROCC'].append(srocc)
        # metrics['PLCC'].append(plcc)
        # metrics['KROCC'].append(krocc)
        # metrics['RMSE'].append(rmse)
        # plot_metrics(metrics)
        # print(f"SROCC={srocc:.4f}, PLCC={plcc:.4f}, KROCC={krocc:.4f}, RMSE={rmse:.4f}")

