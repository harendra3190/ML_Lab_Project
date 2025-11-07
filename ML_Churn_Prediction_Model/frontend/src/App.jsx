import React, { useState } from 'react'
import axios from 'axios'
import './styles.css'

function App() {
  const [formData, setFormData] = useState({
    gender: 'Male',
    SeniorCitizen: 0,
    Partner: 'No',
    Dependents: 'No',
    tenure: 0,
    PhoneService: 'No',
    MultipleLines: 'No',
    InternetService: 'DSL',
    OnlineSecurity: 'No',
    OnlineBackup: 'No',
    DeviceProtection: 'No',
    TechSupport: 'No',
    StreamingTV: 'No',
    StreamingMovies: 'No',
    Contract: 'Month-to-month',
    PaperlessBilling: 'No',
    PaymentMethod: 'Electronic check',
    MonthlyCharges: 0,
    TotalCharges: 0
  })

  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleChange = (e) => {
    const { name, value } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: name === 'SeniorCitizen' || name === 'tenure' || name === 'MonthlyCharges' || name === 'TotalCharges'
        ? (name === 'SeniorCitizen' ? parseInt(value) : parseFloat(value) || 0)
        : value
    }))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await axios.post('http://localhost:5000/predict', formData)
      setResult(response.data)
    } catch (err) {
      setError(err.response?.data?.message || 'Failed to get prediction. Make sure the Flask server is running.')
      console.error('Prediction error:', err)
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setFormData({
      gender: 'Male',
      SeniorCitizen: 0,
      Partner: 'No',
      Dependents: 'No',
      tenure: 0,
      PhoneService: 'No',
      MultipleLines: 'No',
      InternetService: 'DSL',
      OnlineSecurity: 'No',
      OnlineBackup: 'No',
      DeviceProtection: 'No',
      TechSupport: 'No',
      StreamingTV: 'No',
      StreamingMovies: 'No',
      Contract: 'Month-to-month',
      PaperlessBilling: 'No',
      PaymentMethod: 'Electronic check',
      MonthlyCharges: 0,
      TotalCharges: 0
    })
    setResult(null)
    setError(null)
  }

  return (
    <div className="app-container">
      <div className="header">
        <h1>📊 Customer Churn Prediction Dashboard</h1>
        <p>Predict whether a customer will churn using ML models</p>
      </div>

      <div className="main-content">
        <div className="form-container">
          <h2>Customer Information</h2>
          <form onSubmit={handleSubmit}>
            <div className="form-grid">
              <div className="form-group">
                <label>Gender</label>
                <select name="gender" value={formData.gender} onChange={handleChange}>
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                </select>
              </div>

              <div className="form-group">
                <label>Senior Citizen</label>
                <select name="SeniorCitizen" value={formData.SeniorCitizen} onChange={handleChange}>
                  <option value={0}>No</option>
                  <option value={1}>Yes</option>
                </select>
              </div>

              <div className="form-group">
                <label>Partner</label>
                <select name="Partner" value={formData.Partner} onChange={handleChange}>
                  <option value="No">No</option>
                  <option value="Yes">Yes</option>
                </select>
              </div>

              <div className="form-group">
                <label>Dependents</label>
                <select name="Dependents" value={formData.Dependents} onChange={handleChange}>
                  <option value="No">No</option>
                  <option value="Yes">Yes</option>
                </select>
              </div>

              <div className="form-group">
                <label>Tenure (months)</label>
                <input
                  type="number"
                  name="tenure"
                  value={formData.tenure}
                  onChange={handleChange}
                  min="0"
                  max="72"
                />
              </div>

              <div className="form-group">
                <label>Phone Service</label>
                <select name="PhoneService" value={formData.PhoneService} onChange={handleChange}>
                  <option value="No">No</option>
                  <option value="Yes">Yes</option>
                </select>
              </div>

              <div className="form-group">
                <label>Multiple Lines</label>
                <select name="MultipleLines" value={formData.MultipleLines} onChange={handleChange}>
                  <option value="No">No</option>
                  <option value="No phone service">No phone service</option>
                  <option value="Yes">Yes</option>
                </select>
              </div>

              <div className="form-group">
                <label>Internet Service</label>
                <select name="InternetService" value={formData.InternetService} onChange={handleChange}>
                  <option value="DSL">DSL</option>
                  <option value="Fiber optic">Fiber optic</option>
                  <option value="No">No</option>
                </select>
              </div>

              <div className="form-group">
                <label>Online Security</label>
                <select name="OnlineSecurity" value={formData.OnlineSecurity} onChange={handleChange}>
                  <option value="No">No</option>
                  <option value="No internet service">No internet service</option>
                  <option value="Yes">Yes</option>
                </select>
              </div>

              <div className="form-group">
                <label>Online Backup</label>
                <select name="OnlineBackup" value={formData.OnlineBackup} onChange={handleChange}>
                  <option value="No">No</option>
                  <option value="No internet service">No internet service</option>
                  <option value="Yes">Yes</option>
                </select>
              </div>

              <div className="form-group">
                <label>Device Protection</label>
                <select name="DeviceProtection" value={formData.DeviceProtection} onChange={handleChange}>
                  <option value="No">No</option>
                  <option value="No internet service">No internet service</option>
                  <option value="Yes">Yes</option>
                </select>
              </div>

              <div className="form-group">
                <label>Tech Support</label>
                <select name="TechSupport" value={formData.TechSupport} onChange={handleChange}>
                  <option value="No">No</option>
                  <option value="No internet service">No internet service</option>
                  <option value="Yes">Yes</option>
                </select>
              </div>

              <div className="form-group">
                <label>Streaming TV</label>
                <select name="StreamingTV" value={formData.StreamingTV} onChange={handleChange}>
                  <option value="No">No</option>
                  <option value="No internet service">No internet service</option>
                  <option value="Yes">Yes</option>
                </select>
              </div>

              <div className="form-group">
                <label>Streaming Movies</label>
                <select name="StreamingMovies" value={formData.StreamingMovies} onChange={handleChange}>
                  <option value="No">No</option>
                  <option value="No internet service">No internet service</option>
                  <option value="Yes">Yes</option>
                </select>
              </div>

              <div className="form-group">
                <label>Contract</label>
                <select name="Contract" value={formData.Contract} onChange={handleChange}>
                  <option value="Month-to-month">Month-to-month</option>
                  <option value="One year">One year</option>
                  <option value="Two year">Two year</option>
                </select>
              </div>

              <div className="form-group">
                <label>Paperless Billing</label>
                <select name="PaperlessBilling" value={formData.PaperlessBilling} onChange={handleChange}>
                  <option value="No">No</option>
                  <option value="Yes">Yes</option>
                </select>
              </div>

              <div className="form-group">
                <label>Payment Method</label>
                <select name="PaymentMethod" value={formData.PaymentMethod} onChange={handleChange}>
                  <option value="Electronic check">Electronic check</option>
                  <option value="Mailed check">Mailed check</option>
                  <option value="Bank transfer (automatic)">Bank transfer (automatic)</option>
                  <option value="Credit card (automatic)">Credit card (automatic)</option>
                </select>
              </div>

              <div className="form-group">
                <label>Monthly Charges ($)</label>
                <input
                  type="number"
                  name="MonthlyCharges"
                  value={formData.MonthlyCharges}
                  onChange={handleChange}
                  min="0"
                  step="0.01"
                />
              </div>

              <div className="form-group">
                <label>Total Charges ($)</label>
                <input
                  type="number"
                  name="TotalCharges"
                  value={formData.TotalCharges}
                  onChange={handleChange}
                  min="0"
                  step="0.01"
                />
              </div>
            </div>

            <div className="form-actions">
              <button type="submit" disabled={loading} className="btn-submit">
                {loading ? 'Predicting...' : '🔮 Predict Churn'}
              </button>
              <button type="button" onClick={handleReset} className="btn-reset">
                🔄 Reset
              </button>
            </div>
          </form>
        </div>

        {error && (
          <div className="result-container error">
            <h3>❌ Error</h3>
            <p>{error}</p>
          </div>
        )}

        {result && (
          <div className="result-container">
            <h2>Prediction Result</h2>
            <div className={`prediction-card ${result.prediction === 'Yes' ? 'churn' : 'no-churn'}`}>
              <div className="prediction-icon">
                {result.prediction === 'Yes' ? '🔴' : '🟢'}
              </div>
              <div className="prediction-text">
                <h3>
                  {result.prediction === 'Yes' 
                    ? 'Customer likely to churn' 
                    : 'Customer will stay'}
                </h3>
                <p className="confidence">
                  Confidence: {result.confidence.toFixed(2)}%
                </p>
              </div>
            </div>

            <div className="model-details">
              <h3>Model Predictions</h3>
              <div className="model-cards">
                <div className="model-card">
                  <h4>Logistic Regression</h4>
                  <p className="model-prediction">
                    {result.models.logistic_regression.prediction === 'Yes' ? '🔴 Churn' : '🟢 Stay'}
                  </p>
                  <p className="model-probability">
                    Probability: {result.models.logistic_regression.probability.toFixed(2)}%
                  </p>
                </div>
                <div className="model-card">
                  <h4>Decision Tree</h4>
                  <p className="model-prediction">
                    {result.models.decision_tree.prediction === 'Yes' ? '🔴 Churn' : '🟢 Stay'}
                  </p>
                  <p className="model-probability">
                    Probability: {result.models.decision_tree.probability.toFixed(2)}%
                  </p>
                </div>
              </div>
            </div>

            <div className="feature-importance">
              <h3>Top Contributing Features</h3>
              <div className="features-list">
                {result.top_features.map((feature, idx) => (
                  <div key={idx} className="feature-item">
                    <span className="feature-name">{feature.feature}</span>
                    <div className="feature-bar-container">
                      <div 
                        className="feature-bar" 
                        style={{ width: `${feature.importance * 100}%` }}
                      ></div>
                    </div>
                    <span className="feature-value">{(feature.importance * 100).toFixed(2)}%</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="footer">
        <p>ML Churn Prediction Model - Logistic Regression + Decision Tree</p>
      </div>
    </div>
  )
}

export default App

