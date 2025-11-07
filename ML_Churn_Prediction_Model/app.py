from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load models and encoders
print("Loading models...")
try:
    lr_pipeline = joblib.load('model data/ls/churn_pipeline.pkl')
    dt_model = joblib.load('model data/ls/dt_model.pkl')
    
    # Try to load ensemble models (if available)
    try:
        voting_model = joblib.load('model data/ls/voting_model.pkl')
        rf_model = joblib.load('model data/ls/rf_model.pkl')
        use_ensemble = True
        print("✓ Loaded Voting Classifier (best model)")
    except:
        voting_model = None
        rf_model = None
        use_ensemble = False
        print("✓ Using Logistic Regression + Decision Tree")
    
    label_encoders = joblib.load('model data/ls/label_encoders.pkl')
    target_encoder = joblib.load('model data/ls/target_encoder.pkl')
    feature_names = joblib.load('model data/ls/feature_names.pkl')
    print("✓ Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    print("Please run train_model.py first to generate the models")

@app.route('/')
def home():
    return jsonify({
        'message': 'Customer Churn Prediction API',
        'status': 'running',
        'endpoints': {
            '/predict': 'POST - Predict customer churn',
            '/health': 'GET - Health check'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Extract features from request
        features = {
            'gender': data.get('gender', 'Male'),
            'SeniorCitizen': int(data.get('SeniorCitizen', 0)),
            'Partner': data.get('Partner', 'No'),
            'Dependents': data.get('Dependents', 'No'),
            'tenure': int(data.get('tenure', 0)),
            'PhoneService': data.get('PhoneService', 'No'),
            'MultipleLines': data.get('MultipleLines', 'No'),
            'InternetService': data.get('InternetService', 'DSL'),
            'OnlineSecurity': data.get('OnlineSecurity', 'No'),
            'OnlineBackup': data.get('OnlineBackup', 'No'),
            'DeviceProtection': data.get('DeviceProtection', 'No'),
            'TechSupport': data.get('TechSupport', 'No'),
            'StreamingTV': data.get('StreamingTV', 'No'),
            'StreamingMovies': data.get('StreamingMovies', 'No'),
            'Contract': data.get('Contract', 'Month-to-month'),
            'PaperlessBilling': data.get('PaperlessBilling', 'No'),
            'PaymentMethod': data.get('PaymentMethod', 'Electronic check'),
            'MonthlyCharges': float(data.get('MonthlyCharges', 0)),
            'TotalCharges': float(data.get('TotalCharges', 0))
        }
        
        # Create DataFrame
        df = pd.DataFrame([features])
        
        # Handle missing TotalCharges (use median like in training)
        median_total = 1397.475  # Approximate median from training data
        if pd.isna(df['TotalCharges'].iloc[0]) or df['TotalCharges'].iloc[0] == '' or df['TotalCharges'].iloc[0] == 0:
            df['TotalCharges'] = median_total
        
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(median_total)
        
        # Feature Engineering (same as training)
        df_eng = df.copy()
        
        # Calculate average monthly charge per tenure
        df_eng['AvgMonthlyCharge'] = np.where(df_eng['tenure'].iloc[0] > 0,
                                               df_eng['TotalCharges'].iloc[0] / df_eng['tenure'].iloc[0],
                                               df_eng['MonthlyCharges'].iloc[0])
        
        # Create tenure groups
        tenure_val = int(df_eng['tenure'].iloc[0])
        if tenure_val <= 12:
            df_eng['TenureGroup'] = 0
        elif tenure_val <= 24:
            df_eng['TenureGroup'] = 1
        elif tenure_val <= 48:
            df_eng['TenureGroup'] = 2
        else:
            df_eng['TenureGroup'] = 3
        
        # Create monthly charge groups
        monthly_val = df_eng['MonthlyCharges'].iloc[0]
        if monthly_val <= 35:
            df_eng['MonthlyChargeGroup'] = 0
        elif monthly_val <= 70:
            df_eng['MonthlyChargeGroup'] = 1
        elif monthly_val <= 90:
            df_eng['MonthlyChargeGroup'] = 2
        else:
            df_eng['MonthlyChargeGroup'] = 3
        
        # Create ratio features
        df_eng['ChargeRatio'] = np.where(df_eng['MonthlyCharges'].iloc[0] > 0,
                                          df_eng['TotalCharges'].iloc[0] / df_eng['MonthlyCharges'].iloc[0],
                                          0)
        
        # Count of services
        service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        service_count = 0
        for col in service_cols:
            if col in df_eng.columns and df_eng[col].iloc[0] == 'Yes':
                service_count += 1
        df_eng['ServiceCount'] = service_count
        
        # Internet service type as numeric
        internet_map = {'No': 0, 'DSL': 1, 'Fiber optic': 2}
        df_eng['InternetService_numeric'] = internet_map.get(df_eng['InternetService'].iloc[0], 1)
        
        # Contract type as numeric
        contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
        df_eng['Contract_numeric'] = contract_map.get(df_eng['Contract'].iloc[0], 0)
        
        # Payment method risk
        df_eng['PaymentRisk'] = 1 if df_eng['PaymentMethod'].iloc[0] == 'Electronic check' else 0
        
        # Interaction features
        df_eng['Tenure_x_MonthlyCharge'] = df_eng['tenure'].iloc[0] * df_eng['MonthlyCharges'].iloc[0]
        df_eng['Contract_x_Tenure'] = df_eng['Contract_numeric'].iloc[0] * df_eng['tenure'].iloc[0]
        
        # Encode categorical variables
        df_encoded = df_eng.copy()
        for col in label_encoders.keys():
            if col in df_encoded.columns:
                try:
                    # Handle unseen values
                    if df_encoded[col].iloc[0] in label_encoders[col].classes_:
                        df_encoded[col] = label_encoders[col].transform([df_encoded[col].iloc[0]])
                    else:
                        # Use most common class as default
                        df_encoded[col] = label_encoders[col].transform([label_encoders[col].classes_[0]])
                except Exception as e:
                    print(f"Encoding error for {col}: {e}")
                    df_encoded[col] = 0
        
        # Ensure correct order of features
        df_encoded = df_encoded[feature_names]
        
        # Predict with models
        # Logistic Regression (needs scaling)
        lr_pred_proba = lr_pipeline.predict_proba(df_encoded)[0]
        lr_pred = lr_pipeline.predict(df_encoded)[0]
        
        # Decision Tree (no scaling needed)
        dt_pred_proba = dt_model.predict_proba(df_encoded)[0]
        dt_pred = dt_model.predict(df_encoded)[0]
        
        # Use Voting Classifier if available (best model)
        if use_ensemble:
            # Need to scale for voting classifier (it includes LR)
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            # Fit scaler on training data - we'll use a simple approach
            # For prediction, we'll use the voting model if it was trained with scaled data
            try:
                voting_pred_proba = voting_model.predict_proba(df_encoded)[0]
                voting_pred = voting_model.predict(df_encoded)[0]
                voting_churn = target_encoder.inverse_transform([voting_pred])[0]
            except:
                # If voting model needs scaled data, scale it
                df_scaled = lr_pipeline.named_steps['scaler'].transform(df_encoded)
                voting_pred_proba = voting_model.predict_proba(df_scaled)[0]
                voting_pred = voting_model.predict(df_scaled)[0]
                voting_churn = target_encoder.inverse_transform([voting_pred])[0]
            
            # Use voting classifier for final prediction (best model)
            final_prob = voting_pred_proba[1]
            final_prediction = 'Yes' if final_prob >= 0.5 else 'No'
            
            # Get feature importance from Random Forest
            rf_feature_importance = rf_model.feature_importances_
            top_features_idx = np.argsort(rf_feature_importance)[-5:][::-1]
            top_features = [
                {
                    'feature': feature_names[idx],
                    'importance': float(rf_feature_importance[idx])
                }
                for idx in top_features_idx
            ]
        else:
            # Fallback to average of LR and DT
            final_prob = (lr_pred_proba[1] + dt_pred_proba[1]) / 2
            final_prediction = 'Yes' if final_prob >= 0.5 else 'No'
            
            # Get feature importance from Decision Tree
            feature_importance = dt_model.feature_importances_
            top_features_idx = np.argsort(feature_importance)[-5:][::-1]
            top_features = [
                {
                    'feature': feature_names[idx],
                    'importance': float(feature_importance[idx])
                }
                for idx in top_features_idx
            ]
        
        # Decode predictions
        lr_churn = target_encoder.inverse_transform([lr_pred])[0]
        dt_churn = target_encoder.inverse_transform([dt_pred])[0]
        
        response = {
            'prediction': final_prediction,
            'confidence': float(final_prob) * 100,
            'models': {
                'logistic_regression': {
                    'prediction': lr_churn,
                    'probability': float(lr_pred_proba[1]) * 100
                },
                'decision_tree': {
                    'prediction': dt_churn,
                    'probability': float(dt_pred_proba[1]) * 100
                }
            },
            'top_features': top_features,
            'status': 'success'
        }
        
        # Add ensemble model predictions if available
        if use_ensemble:
            response['models']['voting_classifier'] = {
                'prediction': voting_churn,
                'probability': float(voting_pred_proba[1]) * 100
            }
            response['best_model'] = 'Voting Classifier (Ensemble)'
        else:
            response['best_model'] = 'Average of LR + DT'
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

