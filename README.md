# Loan Approval Prediction using Machine Learning

This project explores various machine learning models to predict whether a loan application should be approved or denied, based on customer behavior and demographics. The models were evaluated and compared based on several performance metrics, and extensive data preprocessing and feature engineering techniques were applied.

---


## 📊 Problem Statement

To predict whether a loan will be approved based on a customer's behavioral and demographic attributes, such as marital status, income, car ownership, and professional background. The goal is to assist financial institutions in making data-driven decisions to minimize risk and enhance approval efficiency.

---

## 🗃️ Dataset

The dataset contains over **250,000 records** with the following key features:

- **Numerical**: `income`, `age`, `experience`, `current_job_yrs`, `current_house_yrs`
- **Categorical**: `married/single`, `house_ownership`, `car_ownership`, `profession`, `city`, `state`
- **Target**: `risk_flag` (1 = default, 0 = paid)

> Source: [Kaggle - Loan Prediction Based on Customer Behavior](https://www.kaggle.com/datasets/subhamjain/loan-prediction-based-on-customer-behavior)

---

## 🧠 Methodology

### Phase I: EDA & Feature Engineering
- Cleaned categorical variables and removed noise.
- Addressed imbalance using **RandomOverSampler**.
- Performed **label encoding** and **one-hot encoding**.
- Standardized numerical features using `StandardScaler`.

### Dimensionality Reduction Techniques:
- **Random Forest Feature Importance**
- **Principal Component Analysis (PCA)**
- **Singular Value Decomposition (SVD)**
- **Variance Inflation Factor (VIF)**

### Phase II: Regression Analysis
- Performed **linear regression** on income prediction.
- Used **T-Test** and **F-Test** to evaluate feature significance.
- Applied **backward regression** with Adjusted R² tracking.

### Phase III: Classification Models
Each classifier was optimized using **GridSearchCV** and validated using **Stratified K-Fold Cross-Validation**:

- Decision Tree (with Pre/Post Pruning)
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Support Vector Machine (SVM)
- Random Forest
- Multi-Layer Perceptron (Neural Networks)

### Phase IV: Unsupervised Learning
- **KMeans Clustering** to explore inherent structure in data
- **Apriori Algorithm** for association rule mining

---

## 🏆 Best Model

The **Random Forest Classifier** emerged as the best performing model with:
- Accuracy: ~96%
- High AUC and Specificity

---

## 📈 Visuals & Reports

- Confusion Matrix
- ROC Curves
- PCA & Feature Importance Plots
- Regression Residuals
- KMeans Silhouette & WCSS Plots
- Association Rules Table

(Refer to `Detailed_Report.pdf` for all visuals and plots)

---

## 🛠️ Tech Stack

- Python (Pandas, NumPy, Seaborn, Matplotlib, Scikit-learn, Statsmodels)
- Jupyter Notebooks
- MLxtend (Apriori)
- PrettyTable (Model summaries)
- IPython display tools for rich outputs

---



## 📌 Future Work

- Incorporate additional features like credit score or fraud history
- Try **XGBoost**, **LightGBM** and **ensemble stacking**
- Deploy the best model using a Flask web app

---

## 📜 License

This project is licensed for academic use under the MIT License.

---

## 🙌 Acknowledgments

- Virginia Tech – CS 5804: Introduction to AI  
- Kaggle community for the dataset  
- Scikit-learn and the open-source ML ecosystem



