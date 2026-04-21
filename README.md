# Real-Time Hotel Dynamic Pricing Engine 🏨💰

## Project Overview
The **Real-Time Hotel Dynamic Pricing Engine** is an end-to-end data science application designed to optimize hotel revenue through predictive modeling and real-time price adjustment strategies. By leveraging historical booking data, the engine predicts the baseline **Average Daily Rate (ADR)** and applies dynamic adjustments based on real-time factors like lead time, guest count, seasonality, and cancellation risk.

This project was developed to demonstrate the application of Machine Learning (ML) and explainable AI (XAI) in the hospitality industry, providing a transparent and interactive dashboard for revenue managers.

---

## 🚀 Key Features

### 1. **Predictive ADR Modeling**
- **Algorithm**: Powered by **XGBoost Regressor**, chosen for its high accuracy and ability to handle non-linear relationships in booking patterns.
- **Performance Evaluation**: Real-time tracking of metrics including **MAE**, **RMSE**, and **R² Score**.
- **Feature Importance**: Visual representation of which factors influence pricing the most.

### 2. **Dynamic Pricing Logic**
- **Demand Factor**: Increases price for last-minute or high-occupancy bookings.
- **Seasonality**: Adjusts for peak holiday periods.
- **Risk Mitigation**: Adjusts price based on cancellation probability.

### 3. **Interactive Strategy Comparison**
Compare current rates against Predictive Base Price, Fixed Markup, and Dynamic Pricing strategies.

### 4. **Scenario Testing & Explainability**
- **What-If Simulator**: Test how different inputs affect the recommended price.
- **Explainable AI**: Natural language reasons for every price recommendation.

---

## 🛠️ Technical Architecture

### **Tech Stack**
- **Frontend**: Streamlit
- **Machine Learning**: XGBoost, Scikit-learn
- **Data**: Pandas, NumPy

### **Project Structure**
```text
├── app.py              # Main dashboard
├── train.py            # Model training script
├── model.pkl           # Saved model artifacts
├── requirements.txt    # Dependencies
├── data/               # Datasets
└── src/                # Source logic (cleaning, model, pricing)
```

---

## 📊 Methodology

1.  **Prediction**: XGBoost estimates the base price (ADR).
2.  **Adjustment**:
    - **Demand**: Based on lead time and guests.
    - **Season**: Higher during peak months.
    - **Risk**: Adjusted for cancellation risk.

---

## 💻 How to Run

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Dashboard**:
    ```bash
    streamlit run app.py
    ```
