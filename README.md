# Chest Cancer Classifier

## Overview
This project is a deep learning-based chest cancer classifier developed using transfer learning. The model leverages a pre-trained ImageNet model, which has been fine-tuned to accurately classify chest cancer. The project also incorporates MLOps practices, using MLflow for experiment tracking and DagsHub for data version control.

## Features
- **Transfer Learning**: Utilizes a pre-trained ImageNet model, fine-tuned for chest cancer classification.
- **MLOps Integration**: Implements MLOps practices for streamlined project management and deployment.
- **Experiment Tracking**: MLflow is used for tracking experiments, including model parameters, metrics, and artifacts.
- **Data Version Control**: DagsHub is employed for version control of datasets, ensuring reproducibility and consistency.

## Model Architecture
The classifier is based on a pre-trained convolutional neural network (CNN) from the ImageNet dataset. The final layers of the network were retrained to identify adenocarcinoma chest cancer from medical images, adapting the model for the specific task.

## Getting Started

### Prerequisites
- Python 3.x
- TensorFlow or PyTorch
- MLflow
- DagsHub account

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/chest-cancer-classifier.git
   cd chest-cancer-classifier

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt

3. **Setup MLFlow**:
- Ensure MLflow is properly configured for tracking experiments.
- Optionally, set up a remote MLflow server for centralized tracking.

4. **Setup DagsHub**:
- Connect your DagsHub repository to manage and version your data.

## Workflows
- Update config.yaml
- Update secrets.yaml [Optional]
- Update params.yaml
- Update the entity
- Update the configuration manager in src config
- Update the components
- Update the pipeline
- Update the main.py
- Update the dvc.yaml

## MLflow
- If you're interested in learning how to use MLflow for tracking experiments, managing models, and more, check out this comprehensive [MLflow tutorial](https://www.mlflow.org/docs/latest/tutorial.html).

