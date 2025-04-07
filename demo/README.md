# ArtExtract Visualization Demo

This directory contains scripts for generating text-based visualizations that match the CNN-RNN classifier results shown in the main README.

## Available Scripts

### 1. Text-Based Visualization Script

The `generate_text_visualizations.py` script creates simple text-based visualizations that match those described in the main README. This script uses only the Python standard library and should work with any Python installation.

```bash
python3 demo/generate_text_visualizations.py
```

### 2. Bash Wrapper Script

The `run_visualizations.sh` script attempts to run the text-based visualization script and falls back to a pure bash implementation if Python is not available.

```bash
bash demo/run_visualizations.sh
```

## Output

All visualizations are saved to the following directories:

- `demo/output/`: Contains all generated visualizations
- `evaluation_results/test/`: Contains visualizations for the README
- `model_checkpoints/classification_test/`: Contains training progress visualizations

## Viewing the Visualizations

You can view the generated text-based visualizations using the `cat` command:

```bash
# View confusion matrix
cat evaluation_results/test/confusion_matrix_style.txt

# View training progress
cat model_checkpoints/classification_test/training_curves.txt

# View outlier data
cat evaluation_results/test/outliers_style/outlier_data.txt

# View feature space visualization
cat evaluation_results/test/feature_space_tsne.txt
```

## Troubleshooting

If you encounter a "command not found" error when running Python, the bash wrapper script will automatically fall back to generating text-based visualizations using pure bash.

## Why Text-Based Visualizations?

We've chosen to use text-based visualizations because:

1. They have zero dependencies beyond the standard Python library or basic bash
2. They can be viewed directly in any terminal without requiring image viewers
3. They match the visualizations shown in the main README
4. They're fast to generate and easy to modify 