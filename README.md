# Customer Churn Prediction Model

A complete Machine Learning project for predicting telecom customer churn using Logistic Regression and Decision Tree Classifier with an interactive web dashboard.

## Project Structure

```
telco-churn-project/
├── Telco-Customer-Churn.csv          # Dataset
├── model data/
│   └── ls/
│       ├── churn_pipeline.pkl        # Saved sklearn Pipeline
│       ├── dt_model.pkl              # Decision Tree model
│       ├── label_encoders.pkl        # Label encoders
│       ├── target_encoder.pkl        # Target encoder
│       └── feature_names.pkl         # Feature names
├── notebooks/
│   └── EDA_and_training.ipynb        # Optional notebook
├── train_model.py                    # Training script
├── app.py                            # Flask API
├── requirements.txt                  # Python dependencies
└── frontend/
    ├── package.json
    ├── vite.config.js
    ├── index.html
    └── src/
        ├── App.jsx
        ├── index.jsx
        └── styles.css
```

## Features

- ✅ Data preprocessing with missing value handling
- ✅ Logistic Regression model training
- ✅ Decision Tree Classifier training
- ✅ Model evaluation with accuracy, F1-score, confusion matrix
- ✅ ROC curve visualization
- ✅ Feature importance analysis
- ✅ Interactive React dashboard
- ✅ Flask API for predictions
- ✅ Beautiful, responsive UI

## Setup Instructions

### 1. Install Python Dependencies

```powershell
pip install -r requirements.txt
```

### 2. Train the Models

```powershell
python train_model.py
```

This will:
- Load and preprocess the data
- Train both Logistic Regression and Decision Tree models
- Generate visualizations (ROC curves, confusion matrices, feature importance)
- Save models to `model data/ls/` directory

### 3. Start the Flask API

```powershell
python app.py
```

The API will run on `http://localhost:5000`

### 4. Start the Frontend

Open a new terminal and navigate to the frontend directory:

```powershell
cd frontend
npm install
npm run dev
```

The frontend will run on `http://localhost:3000`

## Usage

1. Open your browser and go to `http://localhost:3000`
2. Fill in the customer information form
3. Click "Predict Churn" button
4. View the prediction results with:
   - Overall prediction (🟢 Stay / 🔴 Churn)
   - Individual model predictions
   - Confidence scores
   - Top contributing features

## Model Performance

The models will be evaluated and compared on:
- **Accuracy**: Overall correctness
- **F1-Score**: Balance between precision and recall
- **ROC AUC**: Area under the ROC curve
- **Confusion Matrix**: Detailed classification results

## Top Features Impacting Churn

The model identifies the most important features:
1. Contract type (Month-to-month vs longer contracts)
2. Tenure (customer loyalty duration)
3. Monthly charges (pricing perception)
4. Total charges (customer lifetime value)
5. Internet service type and support features

## API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /predict` - Predict customer churn

### Example API Request

```json
{
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 24,
  "PhoneService": "Yes",
  "MultipleLines": "Yes",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "Yes",
  "OnlineBackup": "Yes",
  "DeviceProtection": "Yes",
  "TechSupport": "Yes",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Two year",
  "PaperlessBilling": "No",
  "PaymentMethod": "Credit card (automatic)",
  "MonthlyCharges": 89.5,
  "TotalCharges": 2148.0
}
```

## Technologies Used

- **Backend**: Python, Flask, scikit-learn, pandas, numpy
- **Frontend**: React, Vite, Axios
- **Visualization**: Matplotlib, Seaborn

## Notes

- Make sure to train the models first before running the API
- The dataset should be named `Telco-Customer-Churn.csv` in the root directory
- All visualizations are saved in the `model data/` directory

