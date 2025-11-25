# 📊 Indicino Employee Attrition Predictive Analysis

## Project Overview

This project focuses on identifying the root causes of employee attrition at Indicino and building a predictive model to flag employees at high risk of leaving. The goal is to provide the HR Group Head with actionable, data-backed recommendations to reduce the overall turnover rate of **16.12%**.

### Key Findings
1.  **Primary Root Cause:** **Overload Crisis** (Attrition rate triples for employees working overtime).
2.  **Highest Risk Role:** **Sales Representative** (39.76% attrition).
3.  **Model Accuracy:** **87.41%** (Logistic Regression).

---

## 🛠️ Project Structure & Execution

### 1. Data Source
* **File:** `Indicino project.xlsx` (Original dataset containing 1,470 entries and 35 columns)
* **Target Variable (y):** `Attrition` (Binary: 1=Left, 0=Stayed)

### 2. Required Libraries
To run the analysis and model, you need the following Python libraries:
bash 
`pip install pandas numpy scikit-learn matplotlib seaborn`

3. Execution Flow
The project was executed in three primary phases:

| Phase | Description | Key Deliverable |
| :--- | :---: | --- |
| Data Preparation | "Cleaning, Encoding Categorical Features, and Feature Scaling | Fully numerical, scaled feature matrix (X_scaled). |
| Exploratory Data Analysis (EDA) | Deep dive into correlations to find business root causes and identify high-risk groups. | Answers to Case Questions 1, 2, 3, 5, 6. |
| Predictive Modeling | Training the Logistic Regression model and validating EDA findings. | 87.41% Accuracy Score and Feature Coefficients. |


📈 Key Data Insights & Model Validation

A. Root Causes (Validated by Model Coefficients)

| Root Cause | EDA Insight | Model Coefficient (Predictive Power) |
| :--- | :---: | --- |
| Overload Crisis | 30.53% attrition with OverTime. | OverTime: +1.69 (Strongest positive predictor) |
| Stagnation | High correlation between tenure and low pay. | YearsSinceLastPromotion: +1.57 (2nd strongest positive predictor) |
| Retention Factor | 4.37 average years with manager for stayers. | YearsWithCurrManager: −1.16 (Strong negative predictor) |

B. High-Risk Groups
| Group | Attrition Rate | Strategic Implication |
| :--- | :---: | --- |
| Job Role | Sales Representative (39.76%) | Requires immediate, role-specific compensation and workload intervention. |
| Age Band | 18-24 (39.18%) | Indicates a severe failure in early-career engagement and compensation. |

Strategic Recommendations

The following recommendations are derived from both the EDA insights and the predictive validation to target the highest-impact factors.

1. IMMEDIATE WORKLOAD INTERVENTION: Implement an urgent OverTime ban for Sales Representatives and Laboratory Technicians. Rationale: Directly addresses the single biggest attrition driver (OverTime: +1.69).
2. RECOGNITION CYCLE: Mandate a formal promotion/advancement review every $\mathbf{24}$ months for all staff to mitigate the risk posed by long periods without a promotion (YearsSinceLastPromotion: +1.57).
3. STABILIZE MANAGEMENT: Introduce a "New Manager Coaching Program" focused on retention tactics, ensuring managers stay in their roles and quickly build strong relationships to leverage the retention power of YearsWithCurrManager.
4. TARGETED COMPENSATION: Approve a dedicated budget for salary adjustments focused solely on the Sales Representative role and the $18-24$ age band to address the $\mathbf{30\%}$ income gap.

| Detail | Value |
| :--- | --- |
| Model Type | Logistic Regression (Classification) |
| Data Split | 80% Training, 20% Testing (stratify=y) |
| Scaling | MinMaxScaler applied to all numerical features. |
| Final Accuracy | 87.41% |
