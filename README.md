# PyTorch vs Scikit-Learn: Wine Classification Benchmark

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

![Stars](https://img.shields.io/github/stars/yourusername/pytorch-vs-sklearn?style=social)
![Forks](https://img.shields.io/github/forks/yourusername/pytorch-vs-sklearn?style=social)

**A comprehensive benchmark comparing deep learning and traditional machine learning on the UCI Wine Classification dataset**

[View Report](#-results) • [Installation](#-installation) • [Usage](#-usage) • [Findings](#-key-findings)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Quick Start](#-quick-start)
- [Results](#-results)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Key Findings](#-key-findings)
- [Documentation](#-documentation)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

This project provides a rigorous comparison between two popular machine learning approaches:

- **PyTorch**: A deep learning framework for building neural networks
- **Scikit-Learn**: A traditional machine learning library with ensemble methods

Both models are trained on the **UCI Wine Dataset** to classify wine cultivars based on 13 chemical features.

### Dataset Information

| Property | Value |
|----------|-------|
| **Samples** | 178 |
| **Features** | 13 |
| **Classes** | 3 |
| **Train/Test Split** | 80/20 |
| **Preprocessing** | StandardScaler |

### Models Compared

| Framework | Model | Architecture |
|-----------|-------|--------------|
| **PyTorch** | Feedforward Neural Network | 13 → 9 → 10 → 3 |
| **Scikit-Learn** | Random Forest Classifier | 100 estimators |

---

## ⚡ Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/pytorch-vs-sklearn.git
cd pytorch-vs-sklearn

# Install dependencies
pip install -r requirements.txt

# Train both models
python scripts/train_pytorch.py
python scripts/train_sklearn.py

# Compare results
python scripts/compare_models.py
```

---

## 📊 Results

### Performance Metrics

<div align="center">

| Metric | PyTorch | Scikit-Learn | Winner |
|--------|---------|--------------|--------|
| **Accuracy** | 94.44% | 🏆 **100.00%** | Scikit-Learn |
| **Precision** (macro) | 94.44% | 🏆 **100.00%** | Scikit-Learn |
| **Recall** (macro) | 94.44% | 🏆 **100.00%** | Scikit-Learn |
| **Training Time** | ~1.2s | 🏆 **~0.1s** | Scikit-Learn |
| **Inference Time** | ~0.01s | ~0.005s | 🏆 PyTorch |
| **Model Size** | ~2 KB | ~50 KB | 🏆 PyTorch |

### Classification Report - Scikit-Learn

```
              precision    recall  f1-score   support
           0       1.00      1.00      1.00        11
           1       1.00      1.00      1.00        16
           2       1.00      1.00      1.00         9
    accuracy                           1.00        36
   macro avg       1.00      1.00      1.00        36
weighted avg       1.00      1.00      1.00        36
```

### Visualizations

- **Training Loss Curve**: `visualizations/training_loss.png`
- **Model Comparison Chart**: `visualizations/model_comparison.png`
- **Confusion Matrices**: `visualizations/confusion_matrices.png`

</div>

---

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda

### Step-by-Step Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/pytorch-vs-sklearn.git
cd pytorch-vs-sklearn

# 2. Create a virtual environment (recommended)
python -m venv venv

# 3. Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

### Requirements

```
torch>=2.0.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.3.0
```

---

## 📝 Usage

### Train PyTorch Model

```bash
python scripts/train_pytorch.py
```

**Output:**
- Trained model: `saved_models/pytorch_model.pth`
- Results: `results/pytorch_results.json`
- Loss plot: `visualizations/training_loss.png`

### Train Scikit-Learn Model

```bash
python scripts/train_sklearn.py
```

**Output:**
- Trained model: `saved_models/sklearn_model.pkl`
- Results: `results/sklearn_results.json`
- Classification report printed to console

### Compare Models

```bash
python scripts/compare_models.py
```

**Output:**
- Side-by-side comparison printed to console
- Comparison visualization: `visualizations/model_comparison.png`

### Use Trained Models

```python
import torch
from models.pytorch_model import WineNN
from sklearn.preprocessing import StandardScaler
import joblib

# Load PyTorch model
model = WineNN()
model.load_state_dict(torch.load('saved_models/pytorch_model.pth'))
model.eval()

# Load Scikit-Learn model
sklearn_model = joblib.load('saved_models/sklearn_model.pkl')

# Make predictions
with torch.no_grad():
    prediction = model(torch.FloatTensor(scaled_features))
    class_idx = prediction.argmax().item()
```

---

## 📁 Project Structure

```
pytorch-vs-sklearn/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore rules
│
├── data/
│   └── wine_dataset_info.txt         # Dataset documentation
│
├── models/
│   ├── __init__.py
│   ├── pytorch_model.py              # PyTorch NN architecture
│   └── sklearn_model.py              # Scikit-Learn pipeline
│
├── scripts/
│   ├── train_pytorch.py              # Train PyTorch model
│   ├── train_sklearn.py              # Train Scikit-Learn model
│   └── compare_models.py             # Compare both models
│
├── results/
│   ├── pytorch_results.json          # PyTorch metrics
│   ├── sklearn_results.json          # Scikit-Learn metrics
│   └── comparison.json               # Consolidated results
│
├── visualizations/
│   ├── training_loss.png             # Training curve
│   ├── model_comparison.png          # Accuracy comparison
│   └── confusion_matrices.png        # Confusion matrices
│
├── saved_models/
│   ├── pytorch_model.pth             # Trained PyTorch model
│   └── sklearn_model.pkl             # Trained Scikit-Learn model
│
└── notebooks/
    └── analysis.ipynb                # Jupyter notebook analysis
```

---

## 🔍 Key Findings

### 🏆 Scikit-Learn Advantages

✅ **Perfect Classification** - Achieved 100% accuracy on test set  
✅ **Faster Training** - ~10x faster than neural network  
✅ **No Hyperparameter Tuning** - Works well out of the box  
✅ **Interpretability** - Feature importance readily available  
✅ **Smaller Dataset Friendly** - Excels with limited samples (178 total)  
✅ **Production Ready** - Simpler deployment pipeline  

### 🔬 PyTorch Advantages

✅ **Flexibility** - Easy to experiment with custom architectures  
✅ **Scalability** - Better for larger datasets and complex patterns  
✅ **GPU Acceleration** - Can leverage CUDA for speed  
✅ **Research Friendly** - More control over training process  
✅ **Model Size** - Smaller footprint (2 KB vs 50 KB)  
✅ **Faster Inference** - Lower latency for predictions  

### 📈 Insights

1. **Dataset Size Matters**: With only 178 samples, Random Forest generalizes better than a small neural network
2. **No One-Size-Fits-All**: Choose based on problem constraints:
   - Small dataset? → Scikit-Learn
   - Large dataset? → PyTorch/Deep Learning
   - Need interpretability? → Scikit-Learn
   - Need flexibility? → PyTorch
3. **Ensemble Methods**: Traditional ensembles remain highly competitive for structured data

---

## 📚 Documentation

### PyTorch Model Architecture

```
Input Layer (13 features)
    ↓
Dense Layer 1 (13 → 9, ReLU)
    ↓
Dense Layer 2 (9 → 10, ReLU)
    ↓
Output Layer (10 → 3, Softmax via CrossEntropyLoss)
    ↓
Classification (3 classes)
```

### Hyperparameters

**PyTorch:**
- Learning Rate: 0.01
- Optimizer: Adam
- Loss Function: CrossEntropyLoss
- Epochs: 200
- Batch Size: All (no batching)

**Scikit-Learn:**
- n_estimators: 100
- random_state: 41
- Scaler: StandardScaler
- Imputer: Mean strategy

---

## 🚀 Future Improvements

- [ ] Add cross-validation for more robust evaluation
- [ ] Implement hyperparameter tuning (GridSearchCV, Optuna)
- [ ] Add data augmentation techniques
- [ ] Create interactive visualization dashboard
- [ ] Add more baseline models (SVM, Logistic Regression, XGBoost)
- [ ] Implement class imbalance handling
- [ ] Add feature importance analysis
- [ ] Create Docker containerization
- [ ] Add GitHub Actions CI/CD pipeline
- [ ] Expand to other datasets for comparison

---

## 🔗 Related Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Scikit-Learn Documentation](https://scikit-learn.org/)
- [UCI Wine Dataset](https://archive.ics.uci.edu/dataset/109/wine)
- [Deep Learning vs Machine Learning](https://towardsdatascience.com/)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📋 Citation

If you use this project in your research or work, please cite it as:

```bibtex
@software{pytorch_vs_sklearn_2024,
  author = Shlok Nair,
  title = {PyTorch vs Scikit-Learn: Wine Classification Benchmark},
  year = {2024},
  url = {https://github.com/shlok1005/pytorch-vs-sklearn}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to in writing, to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 👤 Author

**Shlok Nair**
- GitHub: https://github.com/shlok1005
- LinkedIn: [(https://linkedin.com/in/shlok-nair](https://www.linkedin.com/in/shlok-nair-050ba6229?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)
- Email: shloknair1024@gmail.com 

---

## ⭐ Show Your Support

Give a ⭐️ if this project helped you! 

---

<div align="center">

**Made with ❤️ by Shlok Nair**

[⬆ Back to top](#pytorch-vs-sklearn-wine-classification-benchmark)

</div>
