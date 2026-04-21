# Real-Time Hotel Dynamic Pricing Engine 🏨💰

## Project Overview
The **Real-Time Hotel Dynamic Pricing Engine** is an end-to-end data science application designed to optimize hotel revenue through predictive modeling and real-time price adjustment strategies. By leveraging historical booking data, the engine predicts the baseline **Average Daily Rate (ADR)** and applies dynamic adjustments based on real-time factors like lead time, guest count, seasonality, and cancellation risk.

This project was developed to demonstrate the application of Machine Learning (ML) and explainable AI (XAI) in the hospitality industry, providing a transparent and interactive dashboard for revenue managers.

---

## 🚀 Key Features

### 1. **Predictive ADR Modeling**
- **Algorithm**: Powered by **XGBoost Regressor**, chosen for its high accuracy and ability to handle non-linear relationships in booking patterns.
- **Performance Evaluation**: Real-time tracking of metrics including **mean absolute error (MAE)**, **Root Mean Squared Error (RMSE)**, and **R² Score**.
- **Feature Importance**: Visual representation of which factors (e.g., lead time, stay length) influence pricing the most.

### 2. **Dynamic Pricing Logic**
The engine uses a multi-factor pricing formula to adjust the predicted base price:
- **Demand Factor**: Increases price automatically for last-minute bookings or large groups.
- **Seasonality**: Adjusts for peak vs. regular holiday seasons.
- **Risk Mitigation**: Factoring in cancellation probability to optimize occupancy.

### 3. **Interactive Strategy Comparison**
- **Base Prediction**: The raw ML estimate.
- **Fixed Markup**: A standard 5% increase over the base.
- **Dynamic Policy**: The intelligence-backed price recommend by the engine.
- **Gap Analysis**: Comparison between actual historical ADR and recommended pricing to identify revenue-capture opportunities.

### 4. **Scenario Testing & Explainability**
- A **"What-If" simulator** allows users to input custom booking parameters (lead time, guests, stay duration) to see real-time price recommendations.
- **Natural Language Recommendations**: The system explains *why* it suggests a price change (e.g., "Recommended to increase price because of high booking demand and peak season effect").

### 5. **Live Booking Simulation**
- A streaming dashboard that simulates a real-time booking environment, visualizing pricing trends, demand scores, and cumulative revenue gain.

---

## 🛠️ Technical Architecture

### **Tech Stack**
- **Frontend/Dashboard**: [Streamlit](https://streamlit.io/)
- **Core Analysis**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Machine Learning**: [XGBoost](https://xgboost.readthedocs.io/), [Scikit-learn](https://scikit-learn.org/)
- **Visualization**: Streamlit Native Charts, Matplotlib

### **Project Structure**
```text
├── app.py              # Main Streamlit dashboard
├── train.py            # Model training script
├── model.pkl           # Saved XGBoost model artifacts
├── data/
│   └── hotel_bookings.csv  # Dataset source
└── src/
    ├── data_cleaning.py    # Preprocessing & feature engineering
    ├── model.py            # ML training & evaluation logic
    └── pricing.py          # Dynamic pricing rule engine
```

---

## 📊 Methodology

1.  **Data Ingestion**: Processes historical hotel booking data, handling missing values and engineering features like `stay_length` and `total_guests`.
2.  **Training**: An XGBoost model is trained to minimize the error between predicted price and actual ADR.
3.  **Real-Time Inference**: When a new booking is simulated, the model predicts a base price.
4.  **Adjustment Engine**:
    - **Demand Score**: Calculated based on lead time (<20 days) and group size.
    - **Seasonality**: Increases price by ~30% during identified peak periods.
    - **Cancellation Risk**: Adjusts price downward slightly for high-risk bookings to ensure competitive positioning.

---

## 💻 How to Run

1.  **Install Dependencies**:
    ```bash
    pip install streamlit pandas xgboost scikit-learn
    ```

2.  **Run the Dashboard**:
    ```bash
    streamlit run app.py
    ```

---

## 🎓 Impact & Learning Outcomes
This project demonstrates proficiency in:
- Building **Robust Data Pipelines** for streaming-like simulations.
- Implementing **Supervised Machine Learning** (Regression) for business forecasting.
- Designing **Explainable AI (XAI)** components to build trust with end-users.
- Developing **Real-time Dashboards** that bridge the gap between technical models and business strategy.
