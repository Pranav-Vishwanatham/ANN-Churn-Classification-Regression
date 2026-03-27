# Customer Churn & Salary Prediction

A machine learning application that predicts customer churn probability and estimated salary using Artificial Neural Networks (ANNs). Built with TensorFlow, Scikit-learn, and Streamlit.

## Features

- **Churn Prediction (Classification)**: Predicts whether a customer will leave using a binary classification ANN model
- **Salary Prediction (Regression)**: Estimates customer salary based on profile attributes using a regression ANN model
- **Interactive UI**: Tab-based Streamlit interface for easy model interaction
- **Hyperparameter Tuning**: Grid search optimization for both classification and regression models
- **TensorBoard Integration**: Training visualization and metrics tracking
- **Model Persistence**: Save and load trained models for inference

## Project Structure

```
.
├── app.py                              # Streamlit web application (main entry point)
├── requirements.txt                    # Python dependencies
├── Churn_Modelling.csv                # Dataset
├── experiments.ipynb                   # Initial ANN experimentation & training
├── hyperparametertuningann.ipynb       # Classification model hyperparameter tuning
├── hyperparametertuningrgresn.ipynb    # Regression model hyperparameter tuning
├── salaryregression.ipynb              # Regression model training pipeline
├── prediction.ipynb                    # Prediction testing notebook
├── churn_classification_model.h5       # Trained classification model
├── salary_regression_model.h5          # Trained regression model
├── label_encoder_gender.pkl            # Gender encoder
├── onehot_encoder_geo.pkl              # Geography one-hot encoder
├── scaler.pkl                          # Features scaler for classification
├── scaler_reg.pkl                      # Features scaler for regression
├── model.h5                            # Additional model checkpoint
└── logs/                               # TensorBoard logs directory
    └── fit/                            # Training event logs
```

## Installation

### Prerequisites

- Python 3.9+
- pip or conda

### Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/customer-churn-salary-predictor.git
   cd customer-churn-salary-predictor
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

The app will open at `http://localhost:8501` with two tabs:

#### Tab 1: Churn Prediction

- Input customer attributes (Credit Score, Age, Balance, Tenure, etc.)
- Get probability of customer churn (0-1)
- Interpretation: > 0.5 = likely to churn, ≤ 0.5 = likely to stay

#### Tab 2: Salary Prediction

- Input customer attributes (excluding salary)
- Select customer status (Exited: 0 = Active, 1 = Churned)
- Get estimated salary prediction

### Training Models

#### Classification Model (Churn Prediction)

```bash
jupyter notebook experiments.ipynb
```

Then run hyperparameter tuning:

```bash
jupyter notebook hyperparametertuningann.ipynb
```

#### Regression Model (Salary Prediction)

```bash
jupyter notebook salaryregression.ipynb
```

Then run hyperparameter tuning:

```bash
jupyter notebook hyperparametertuningrgresn.ipynb
```

### Viewing Training Metrics (TensorBoard)

```bash
tensorboard --logdir logs/fit/
```

Open `http://localhost:6006` in your browser to visualize training curves.

## Model Architecture

### Classification Model (Churn)

- Input Layer: 11 features (after encoding & one-hot)
- Hidden Layer 1: 64 neurons + ReLU
- Hidden Layer 2: 32 neurons + ReLU
- Output Layer: 1 neuron + Sigmoid
- Loss: Binary Crossentropy
- Optimizer: Adam (lr=0.01)

### Regression Model (Salary)

- Input Layer: 12 features (including Exited status)
- Hidden Layer 1: 64 neurons + ReLU
- Hidden Layer 2: 32 neurons + ReLU
- Output Layer: 1 neuron + Linear
- Loss: Mean Squared Error (MSE)
- Optimizer: Adam

## Data Preprocessing

1. **Feature Engineering**
   - Removed ID columns (RowNumber, CustomerId, Surname)
   - Label encoded categorical: Gender
   - One-hot encoded categorical: Geography
   - Numerical features: StandardScaler normalization

2. **Train-Test Split**
   - 80-20 split with `random_state=42`
   - Stratified splitting for classification

3. **Feature Sets**
   - Classification: All features except target (Exited)
   - Regression: All features except target (EstimatedSalary) but includes Exited

## Dependencies

Key packages (see `requirements.txt` for full list):

- `tensorflow==2.15.0` - Deep learning framework
- `numpy==1.26.4` - Numerical computing
- `scikit-learn==1.3.0` - Machine learning utilities
- `scikeras==0.12.0` - Scikit-learn wrapper for Keras
- `streamlit` - Web app framework
- `pandas` - Data manipulation
- `tensorboard` - Training visualization

## Hyperparameter Tuning

The project includes GridSearchCV optimization for:

**Classification**

- Neurons: [16, 32, 64, 128]
- Layers: [1, 2]
- Epochs: [50, 100]
- CV: 3-fold cross-validation

**Regression**

- Neurons: [16, 32, 64, 128]
- Layers: [1, 2]
- Epochs: [50, 100]
- CV: 3-fold cross-validation

## Results & Performance

### Classification (Churn)

- Metrics: Accuracy, Precision, Recall, F1-Score
- Cross-validation with early stopping

### Regression (Salary)

- Metrics: MAE, MSE, RMSE, R²
- Cross-validation with early stopping

(Add actual performance metrics after training)

## Common Issues & Solutions

### ImportError: `_ARRAY_API not found`

- **Cause**: TensorFlow 2.16+ incompatible with NumPy 2.0+
- **Solution**: Use TF 2.15.0 and NumPy <2.0
  ```bash
  pip install tensorflow==2.15.0 "numpy<2.0"
  ```

### Feature Names Mismatch in Scaler

- **Cause**: Training features don't match app input features
- **Solution**: Ensure app features match training data preprocessing, especially `Exited` inclusion

### Port Already in Use (TensorBoard/Streamlit)

- **Solution**: Specify custom port
  ```bash
  tensorboard --logdir logs/fit/ --port=6008
  streamlit run app.py --server.port 8502
  ```

## Future Improvements

- [ ] Add model explainability (SHAP, LIME)
- [ ] Implement ensemble methods (Random Forest, Gradient Boosting)
- [ ] Add data validation & error handling in app
- [ ] Deploy to cloud (AWS, GCP, Heroku)
- [ ] Create REST API with FastAPI
- [ ] Add confidence intervals for predictions
- [ ] Implement A/B testing framework
- [ ] Add feature importance visualization

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/your-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Author

Pranav

## Contact

For questions or feedback, please open an issue in the repository.

## Acknowledgments

- Dataset: [Churn Modelling Dataset](https://www.kaggle.com/)
- TensorFlow & Keras documentation
- Streamlit community
