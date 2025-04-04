# MNIST Multi-Model Training and Ensemble

This project implements a comprehensive pipeline for training the MNIST digit classification dataset using a variety of classical machine learning and deep learning models. It also combines select models using an ensemble Voting Classifier to improve performance and robustness.

## Models Implemented

- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Logistic Regression
- Decision Tree
- Random Forest
- Multi-Layer Perceptron (MLP)
- Artificial Neural Network (ANN) using Keras
- Convolutional Neural Network (CNN)
- Recurrent Neural Network (RNN) using LSTM
- Voting Classifier (Ensemble of selected classical models)

## Dataset

The [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset of handwritten digits is used, loaded via `fetch_openml`.

- 70,000 grayscale images of digits (28x28)
- 10 classes (digits 0–9)

## Project Structure

```
.
├── MNIST_Multi_Model_Training_and_Ensemble.ipynb
├── README.md
```

Each model is trained independently with detailed print statements and error handling. Memory management is handled using Python's `gc` module to prevent out-of-memory issues.

## Results Summary

| Model           | Accuracy     |
|----------------|--------------|
| KNN            | 97.13%       |
| SVM            | 95.40% (on subset) |
| Logistic Reg.  | 92.03%       |
| Decision Tree  | 87.15%       |
| Random Forest  | 96.40%       |
| MLP            | 96.00%       |
| ANN (Keras)    | 96.58%       |
| CNN            | 98.64%       |
| RNN (LSTM)     | 97.48%       |
| Voting Clf     | 96.03%       |

## Requirements

Make sure to install the necessary packages:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn tensorflow
```

## How to Use

1. Clone the repo:
    ```bash
    git clone https://github.com/kssrikar4/MNIST_Multi_Model_Training_and_Ensemble.git
    cd mnist-multi-model-ensemble
    ```

2. Run the notebook:
    ```bash
    jupyter notebook MNIST_Multi_Model_Training_and_Ensemble.ipynb
    ```

## Notes

- SVM is trained on a subset of the data for performance reasons.
- Each model is cleared from memory after evaluation.
- VotingClassifier combines KNN, Logistic Regression, and Random Forest.

## Author

– [@kssrikar4](https://github.com/kssrikar4)
