# Directory Structure 
VQA/

* args_config.py          : Configuration for command-line arguments
* extract_features.py     : Script to extract features from videos
* load_data.py            : Utility to load video frames
* main.py                 : Main script to run feature extraction, training, and testing
* models.py               : Model definitions (ResNet, BiLSTM)
* parser.py               : Custom argument parser
* train.py                : Script to train the model
* test.py                 : Script to test the model
* utils.py                : Utility functions (metrics, plotting)
* VQA_datasets.py         : Dataset classes for VQA
* README.md               : This file

# How to run : 

## Feature Extraction
Extract features from the video dataset using the pre-trained ResNet model combined to Faster-RCNN:


`python main.py --run_extraction --batch_size 4
--train_data_path /path/to/train_videos --train_labels_path /path/to/train_labels.csv 
--val_data_path /path/to/val_videos --val_labels_path /path/to/val_labels.csv 
--test_data_path /path/to/test_videos
--train_data_dir /path/to_save/train_features 
--val_data_dir /path/to_save/val_features 
--test_data_dir /path/to_save/test_features 
`

## Training 
Train the BiLSTM model on the extracted features:

`python main.py --run_training
--train_data_dir /path/to/train_features 
--val_data_dir /path/to/val_features 
--epochs 100 --batch_size 64 --learning_rate 1e-4`

## Testing 
Test the trained model on the test dataset:

`python main.py --run_testing
 --test_data_dir /path/to/test_features --test_example_csv /path/to/csvExample
 --batch_size 64`