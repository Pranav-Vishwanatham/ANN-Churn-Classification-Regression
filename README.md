# ANN Churn Classification & Regression

A machine learning application that predicts customer churn probability and estimated salary using Artificial Neural Networks (ANNs). Built with TensorFlow, Scikit-learn, and Streamlit.

## Features

- **Churn Prediction (Classification)**: Predicts whether a customer will leave using a binary classification ANN model
- **Salary Prediction (Regression)**: Estimates customer salary based on profile attributes using a regression ANN model
- **Interactive UI**: Tab-based Streamlit interface for easy model interaction and real-time predictions
- **Hyperparameter Tuning**: Grid search optimization with cross-validation for both classification and regression models
- **TensorBoard Integration**: Training visualization and metrics tracking with event logs
- **Model Persistence**: Save and load trained models for inference and deployment
- **Feature Engineering**: Comprehensive data preprocessing including encoding and scaling

## Project Structure

```
.
├── app.py                              # Streamlit web application (main entry point)
├── requirements.txt                    # Python dependencies
├── README.md                           # Project documentation
├── .gitignore                          # Git ignore configurations
├── Churn_Modelling.csv                 # Customer dataset with churn and salary labels
├── churn_classification.ipynb          # Initial ANN experimentation & classification model training
├── hyperparametertuningcls.ipynb       # Classification model hyperparameter tuning with GridSearchCV
├── hyperparametertuningrgresn.ipynb    # Regression model hyperparameter tuning with GridSearchCV
├── salaryregression.ipynb              # Regression model training and evaluation pipeline
├── prediction.ipynb                    # Model prediction testing and validation notebook
├── (Model files - ignored in git)
│   ├── churn_classification_model.h5   # Trained classification model (ANN)
│   ├── salary_regression_model.h5      # Trained regression model (ANN)
│   ├── label_encoder_gender.pkl        # Gender label encoder
│   ├── onehot_encoder_geo.pkl          # Geography one-hot encoder
│   ├── scaler_classification.pkl                      # StandardScaler for classification features
│   └── scaler_reg_regression.pkl                  # StandardScaler for regression features
└── (TensorBoard logs - ignored in git)
    └── logs/fit/                       # TensorBoard event logs for training metrics
```

## Installation

### Prerequisites

- Python 3.9+
- pip or conda
- Git

### Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/Pranav-Vishwanatham/ANN-Churn-Classification-Regression
   cd ANN-Churn-Classification-Regression
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Web Application

```bash
streamlit run app.py
```

The application will open with two interactive tabs:

#### Tab 1: Churn Prediction

- Input customer attributes:
  - Credit Score, Gender, Age, Geography
  - Tenure, Balance, Number of Products
  - Has Credit Card, Is Active Member
  - Estimated Salary
- Output: Churn probability (0-1 scale)
- Interpretation: > 0.5 = likely to churn, ≤ 0.5 = likely to stay

#### Tab 2: Salary Prediction

- Input customer attributes (same as above except salary)
- Select customer exit status: Exited (0 = Active, 1 = Churned)
- Output: Estimated salary prediction in currency format

### Training Models

#### 1. Classification Model (Churn Prediction)

**Initial Training & Experimentation:**

```bash
jupyter notebook churn_classification.ipynb
```

**Hyperparameter Tuning:**

```bash
jupyter notebook hyperparametertuningcls.ipynb
```

#### 2. Regression Model (Salary Prediction)

**Model Training & Evaluation:**

```bash
jupyter notebook salaryregression.ipynb
```

**Hyperparameter Tuning:**

```bash
jupyter notebook hyperparametertuningrgresn.ipynb
```

#### 3. Prediction Testing

**Test predictions and validate model performance:**

```bash
jupyter notebook prediction.ipynb
```

### Viewing Training Metrics with TensorBoard

After training, visualize your model training curves and metrics:

```bash
tensorboard --logdir logs/fit/
```

Open `http://localhost:6006` in your browser to see:

- Loss curves (training vs validation)
- Accuracy metrics
- Histograms of weights
- Other training statistics

## Model Architecture

### Classification Model (Churn Prediction)

- **Input Layer**: 11 features (CreditScore, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Geography_encoded)
- **Hidden Layer 1**: 64 neurons + ReLU activation
- **Output Layer**: 1 neuron + Sigmoid activation
- **Loss Function**: Binary Crossentropy
- **Optimizer**: Adam (learning_rate=0.0001)
- **Metrics**: Accuracy
- **Callbacks**: TensorBoard logging

### Regression Model (Salary Prediction)

- **Input Layer**: 10 features (CreditScore, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, Geography_encoded)
- **Hidden Layer 1**: 128 neurons + ReLU activation
- **Hidden Layer 2**: 128 neurons + ReLU activation
- **Output Layer**: 1 neuron + Linear activation (default)
- **Loss Function**: Mean Absolute Error (MAE)
- **Optimizer**: Adam (learning_rate=0.0001)
- **Metrics**: MAE (Mean Absolute Error)
- **Callbacks**: TensorBoard logging

## Data Preprocessing Pipeline

### Step 1: Data Loading & Cleaning

- Load `Churn_Modelling.csv` (10,000 customer records)
- Drop irrelevant columns: RowNumber, CustomerId, Surname
- Handle missing values (none present in this dataset)

### Step 2: Feature Engineering

- **Label Encoding**: Gender → {0, 1}
- **One-Hot Encoding**: Geography → France, Germany, Spain (3 binary columns)
- **Normalization**: StandardScaler on all numerical features

### Step 3: Train-Test Split

- 80-20 split with `random_state=42`
- Classification: Stratified split to preserve class distribution
- Both splits use same random state for reproducibility

### Step 4: Feature Organization

**Classification (Churn):**

- Input: 11 features (all except Exited)
- Target: Exited (binary: 0/1)

**Regression (Salary):**

- Input: 12 features (all except EstimatedSalary, includes Exited)
- Target: EstimatedSalary (continuous values)

## Dependencies & Requirements

Key packages (complete list in `requirements.txt`):

```
tensorflow==2.15.0        # Deep learning framework
tensorflow-macos==2.15.0  # macOS specific version
tensorflow-metal          # Metal GPU acceleration (macOS)
keras==2.15.0            # Keras API
numpy==1.26.4            # Numerical computing (MUST be <2.0)
scikit-learn==1.3.0      # ML utilities, preprocessing, GridSearchCV
scikeras==0.12.0         # Scikit-learn wrapper for Keras models
pandas                    # Data manipulation and analysis
streamlit                 # Web application framework
tensorboard              # Training visualization and monitoring
matplotlib               # Data visualization
```

**Important**: NumPy must be <2.0 due to TensorFlow 2.15 compatibility. Using NumPy 2.0+ will cause `_ARRAY_API` errors.

## Hyperparameter Tuning with GridSearchCV

Both models use GridSearchCV for systematic hyperparameter optimization:

### Classification Tuning

```python
param_grid = {
    'neurons': [16, 32, 64, 128],      # Hidden layer size
    'layers': [1, 2],                  # Number of hidden layers
    'epochs': [50, 100]                # Training epochs
}
# Total: 4 × 2 × 2 = 16 parameter combinations
# With 3-fold CV: 16 × 3 = 48 model fits
```

### Regression Tuning

Same grid search parameters as classification:

```python
param_grid = {
    'neurons': [16, 32, 64, 128],
    'layers': [1, 2],
    'epochs': [50, 100]
}
# Total: 16 combinations × 3 folds = 48 fits
```

Best models are selected based on CV scores and saved for deployment.

## Model Performance & Evaluation Metrics

### Classification (Churn Prediction)

- **Accuracy**: Overall proportion of correct predictions

### Regression (Salary Prediction)

- **MAE**: Average absolute prediction error in currency units

## Project Files Guide

| File                               | Purpose                                  | Type               |
| ---------------------------------- | ---------------------------------------- | ------------------ |
| `app.py`                           | Streamlit web application                | Production Code    |
| `churn_classification.ipynb`       | Initial classification model development | Notebook           |
| `hyperparametertuningcls.ipynb`    | GridSearchCV for classification tuning   | Notebook           |
| `salaryregression.ipynb`           | Regression model development             | Notebook           |
| `hyperparametertuningrgresn.ipynb` | GridSearchCV for regression tuning       | Notebook           |
| `prediction.ipynb`                 | Model testing and validation             | Notebook           |
| `Churn_Modelling.csv`              | Training dataset                         | Data (10K records) |
| `requirements.txt`                 | Python package dependencies              | Config             |
| `README.md`                        | Project documentation                    | Documentation      |
| `.gitignore`                       | Git ignore patterns                      | Config             |

## Common Issues & Troubleshooting

### Issue 1: `ImportError: _ARRAY_API not found`

**Cause**: Incompatible TensorFlow/NumPy versions
**Solution**:

```bash
pip install tensorflow==2.15.0 "numpy<2.0" scikit-learn==1.3.0
```

### Issue 2: Feature Names Mismatch Error

**Cause**: Scaler trained on different features than prediction input
**Solution**: Ensure training and prediction use identical feature sets

- Classification: Exited is target, NOT input
- Regression: Include Exited as input feature

### Issue 3: Port Already in Use

**Cause**: Previous Streamlit/TensorBoard process still running
**Solution**:

```bash
# Use different ports
tensorboard --logdir logs/fit/ --port=6008
streamlit run app.py --server.port=8502
```

### Issue 4: CUDA Not Found (GPU)

**Note**: On macOS with Apple Silicon, Metal acceleration is used automatically

- No additional setup required
- TensorFlow detects and uses Metal GPU acceleration

### Issue 5: Scikeras KerasClassifier for Regression

**Cause**: Using KerasClassifier for continuous targets
**Solution**: Use KerasRegressor for regression tasks

```python
from scikeras.wrappers import KerasRegressor
model = KerasRegressor(build_fn=create_model, ...)
```

## Future Enhancements & Roadmap

- [ ] Add SHAP/LIME interpretability for model decisions
- [ ] Implement ensemble methods (Random Forest, XGBoost, Voting)
- [ ] Add comprehensive input validation and error handling
- [ ] Deploy on cloud (AWS SageMaker, Google Cloud AI, Azure ML Service)
- [ ] Develop FastAPI backend for REST API integration
- [ ] Add prediction confidence intervals and uncertainty estimates
- [ ] Implement A/B testing framework for model variants
- [ ] Feature importance analysis with visualization
- [ ] Docker containerization for easy deployment
- [ ] Batch prediction capability for CSV files

## Contributing Guidelines

Contributions welcome! Please follow these steps:

1. Fork this repository
2. Create a feature branch (`git checkout -b feature/your-feature-name`)
3. Make your changes with clear, descriptive commit messages
4. Test your changes thoroughly
5. Push to your fork (`git push origin feature/your-feature-name`)
6. Open a Pull Request with detailed description

## Project License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for full details.

## Contact & Support

Have questions or found a bug?

- **GitHub Issues**: Open an issue on the repository
- **Email**: pranav@example.com
- **Discussion**: Start a discussion in the repository

## Acknowledgments & Resources

- **Dataset**: [Kaggle Churn Modelling](https://www.kaggle.com/shrutimechlearn/churn-modelling)
- **Framework Docs**:
  - TensorFlow: https://www.tensorflow.org/
  - Streamlit: https://docs.streamlit.io/
  - Scikit-learn: https://scikit-learn.org/
  - SciKeras: https://adriangb.com/scikeras/
- **Community**: Thanks to the open-source ML community for inspiration and tools
