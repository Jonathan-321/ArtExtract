# Scripts

This directory contains the training and evaluation scripts for the ArtExtract CNN-RNN classifier.

## Contents

- `train_cnn_rnn_classifier.py`: Main script for training the CNN-RNN classifier
- `evaluate_cnn_rnn_classifier.py`: Main script for evaluating the CNN-RNN classifier

## How to Run

You can run these scripts directly from the root directory using the wrapper scripts:

```bash
# Train the model
./train.py --data_dir data/test_dataset --batch_size 4 --num_epochs 5 --pretrained --test_mode --backbone resnet18 --save_dir model_checkpoints/classification_test

# Evaluate the model
./evaluate.py --data_dir data/test_dataset --checkpoint model_checkpoints/classification_test/best_style_model.pth --output_dir evaluation_results/test --test_mode --backbone resnet18
```

## Script Details

### Train CNN-RNN Classifier

The training script (`train_cnn_rnn_classifier.py`) trains a CNN-RNN model for classifying art attributes like style, artist, and genre.

Key parameters:
- `--data_dir`: Path to the dataset directory
- `--attributes`: Art attributes to classify (e.g., style, artist, genre)
- `--backbone`: CNN backbone architecture (e.g., resnet18, resnet50)
- `--num_epochs`: Number of epochs to train
- `--save_dir`: Directory to save checkpoints
- `--test_mode`: Use test dataset instead of full WikiArt dataset

### Evaluate CNN-RNN Classifier

The evaluation script (`evaluate_cnn_rnn_classifier.py`) evaluates a trained CNN-RNN model on the test set and identifies outliers.

Key parameters:
- `--data_dir`: Path to the dataset directory
- `--checkpoint`: Path to model checkpoint
- `--output_dir`: Directory to save evaluation results
- `--outlier_method`: Method for outlier detection
- `--num_outliers`: Number of outliers to visualize
- `--test_mode`: Use test dataset instead of full WikiArt dataset 