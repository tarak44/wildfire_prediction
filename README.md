🔥 Wildfire Prediction Project

A deep learning project for predicting wildfires from satellite images using PyTorch.

📋 Project Overview

This project implements a CNN-based classifier to distinguish between images containing wildfires and those without. The model is trained on a dataset of satellite images and achieves high accuracy in wildfire detection.

🗂️ Project Structure
Wildfire_Prediction/
├── train/                     # Training dataset
│   ├── wildfire/             # Wildfire images
│   └── nowildfire/           # Non-wildfire images
├── valid/                     # Validation dataset
│   ├── wildfire/
│   └── nowildfire/
├── test/                      # Test dataset
│   ├── wildfire/
│   └── nowildfire/
├── SOURCECODE.ipynb            # Main Jupyter notebook
├── wildfire_prediction_fixed.py  # Training script
├── run_wildfire_prediction.py    # Command-line interface
├── config.py                  # Configuration file
├── requirements.txt           # Python dependencies
└── README.md                  # This file

🚀 Quick Start
1. Install Dependencies
pip install -r requirements.txt

2. Prepare Dataset

Ensure your dataset is organized as follows:

train/
├── wildfire/
└── nowildfire/
valid/
├── wildfire/
└── nowildfire/
test/
├── wildfire/
└── nowildfire/

3. Run Training

Option A: Command Line Interface

python run_wildfire_prediction.py --train


Option B: Jupyter Notebook

jupyter notebook SOURCECODE.ipynb


Option C: Direct Script Execution

python wildfire_prediction_fixed.py

4. Run Inference
python run_wildfire_prediction.py --inference --image path/to/image.jpg

🔧 Configuration

Edit config.py to modify:

Dataset paths

Model hyperparameters

Training settings

Data augmentation parameters

📊 Model Architectures

WildfireCNN – Custom CNN with 4 convolutional layers.

WildfireResNet – Pre-trained ResNet18 with a custom classifier layer.

📈 Key Features

✅ Robust error handling for corrupted images

✅ Data augmentation for better generalization

✅ Comprehensive evaluation metrics

✅ Model checkpointing and saving

✅ Command-line interface

✅ Detailed logging and progress tracking

✅ Visualization of training curves and results

🎯 Performance Metrics

The model reports:

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

Classification Report

🛠️ Usage Examples

Check Dependencies and Dataset

python run_wildfire_prediction.py --check


Train Model

python run_wildfire_prediction.py --train


Run Inference

python run_wildfire_prediction.py --inference --image test_image.jpg

📝 Notes

The model uses ImageNet normalization values.

Training is optimized for CPU/GPU compatibility.

Model checkpoints are automatically saved.

All results are logged with timestamps.

🤝 Contributing

Feel free to submit issues and enhancement requests!

📄 License

This project is open source and available under the MIT License.