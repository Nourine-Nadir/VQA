from utils import plot_metrics, calculate_metrics, all_metrics
import torch.cuda as cuda
import pandas as pd
from models import *
from VQA_datasets import FeatureDataset
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from torch.utils.data import ConcatDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





def train(config):
    # Load feature datasets
    train_dataset = FeatureDataset(data_dir= config.train_data_dir)

    val_dataset = FeatureDataset(data_dir= config.val_data_dir)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    combined_dataset = ConcatDataset([train_dataset, val_dataset])
    combined_loader = DataLoader(combined_dataset, batch_size=config.batch_size, shuffle=True)

    # Initialize BiSLTM model
    model = BiLSTMForRegression(

        lr=config.learning_rate,

    ).to(DEVICE)
    print(model)

    truth_dir = '/media/nadir/SSD/VQA Data/KVQ/Validation_site/Validation'
    truth_file = os.path.join(truth_dir, "truth.csv")
    submission_answer_file = 'prediction.csv'
    metrics = {
        'SROCC': [],
        'PLCC': [],
        'KROCC': [],
        'SCORE': [],
    }
    best_score = 0
    # Training loop
    for epoch in range(config.epochs):
        model.train()
        for features, mos in tqdm(combined_loader, desc=f"Epoch {epoch + 1}"):
            features = features.to(DEVICE)
            mos = mos.to(DEVICE)

            features1, features2 = features[:len(features) // 2].to(DEVICE), features[len(features) // 2:].to(DEVICE)
            mos1, mos2 = mos[:len(mos) // 2].to(DEVICE), mos[len(mos) // 2:].to(DEVICE)

            model.optimizer.zero_grad()


            predicted_a = model(features1)
            predicted_b = model(features2)
            mse_loss = model.mse_loss_fn(predicted_a.squeeze(), mos1) \
                       + model.mse_loss_fn(predicted_b.squeeze(), mos2)
            ranking_loss = model.ranking_loss_fn(predicted_a, predicted_b, mos1, mos2)

            loss = 0.8 * mse_loss + 0.2 * ranking_loss
            loss.backward()
            model.optimizer.step()

            # Validation loop
            runtimes = []
        # Create CUDA events for timing
        start_event = cuda.Event(enable_timing=True)
        end_event = cuda.Event(enable_timing=True)
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            running_loss = 0.0
            y_pred = []
            for features, mos in tqdm(val_loader, desc="Validation"):
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
            print(f"Total runtime for {len(runtimes)} batches: {sum(runtimes):.4f} seconds")

            outputs_df = pd.read_csv('prediction_.csv')
            outputs_df['score'] = np.squeeze(np.array(y_pred).astype(np.float32))
            outputs_df.set_index('filename', inplace=True)

            outputs_df.to_csv('prediction.csv')
            epoch_loss = running_loss / len(val_dataset)
        print(f'epoch loss {epoch_loss}')
        model.lr_decay(epoch_loss)
        print(f'LR : {model.get_lr()}')

        # Calculate metrics
        srocc, plcc, krocc, rmse = calculate_metrics(val_labels, val_preds)

        score, SROCC, PLCC, acc_non_source, acc_source = all_metrics(submission_answer_file, truth_file, truth_dir)

        print(
            f'score : {score:.4f}\nSROCC : {SROCC:.4f}\nPLCC : {PLCC:.4f}\nacc_non_source : {acc_non_source:.4f}\nacc_source : {acc_source:.4f}\n')

        print(f'srocc {srocc} ')
        if score > best_score:
            model.save_model(path='saved_models/best_BiLSTM')
            best_score = score
            print('new best model saved with score = ', score)

        model.save_model(path='saved_models/last_BiLSTM')

        metrics['SROCC'].append(srocc)
        metrics['PLCC'].append(plcc)
        metrics['KROCC'].append(krocc)
        metrics['SCORE'].append(score)
        plot_metrics(metrics)
        print(
            f"Epoch {epoch + 1}: SROCC={srocc:.4f}, PLCC={plcc:.4f}, KROCC={krocc:.4f}, RMSE={rmse:.4f}, SCORE={score:.4f}")
