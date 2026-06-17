# 📦 E-Commerce Shipping Prediction using Machine Learning

A Machine Learning project that predicts whether an e-commerce shipment will reach customers on time.

This project performs data preprocessing, exploratory data analysis, visualization, model training, and performance evaluation by comparing multiple classification algorithms.

---

## 🚀 Project Overview

The goal of this project is to predict the target variable:

**Reached.on.Time_Y.N**

- `1` → Shipment reached on time
- `0` → Shipment was delayed

The project compares the performance of multiple machine learning algorithms to determine which model is most effective for shipment prediction.

---

## 🎯 Objectives

- Analyze e-commerce shipping data.
- Preprocess and clean the dataset.
- Encode categorical variables.
- Visualize feature relationships.
- Train multiple machine learning models.
- Compare model performance.
- Evaluate prediction accuracy using various metrics.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## 📂 Dataset

**Dataset:** E-commerce Shipping Dataset

The dataset contains shipping-related information such as:

- Warehouse Block
- Mode of Shipment
- Customer Care Calls
- Customer Ratings
- Product Importance
- Gender
- Prior Purchases
- Cost of Product
- Discount Offered
- Weight in grams

### 🎯 Target Variable

`Reached.on.Time_Y.N`

| Value | Meaning |
|-------|---------|
| 1 | Reached on time |
| 0 | Delayed |

---

## ⚙️ Project Workflow

### 1️⃣ Data Loading

The dataset is loaded using Pandas.

### 2️⃣ Data Preprocessing

- Removed the `ID` column (non-predictive feature)
- Encoded categorical variables using `LabelEncoder`

### 3️⃣ Data Visualization

Generated:

- Class distribution plot
- Correlation heatmap

### 4️⃣ Data Splitting

Split the dataset into:

- 70% Training Data
- 30% Testing Data

using `train_test_split()`.

### 5️⃣ Model Training

Trained multiple machine learning models.

### 6️⃣ Model Evaluation

Compared model performance using several evaluation metrics.

---

## Machine Learning Models Used

### K-Nearest Neighbors (KNN)

- Uses `MinMaxScaler`
- Distance-based classification algorithm

### Logistic Regression

- Uses `StandardScaler`
- Linear classification model

### Decision Tree

- Uses `StandardScaler`
- Tree-based classification model

### Neural Network (MLP Classifier)

- Uses `StandardScaler`
- Multi-layer perceptron neural network

---

## 📊 Evaluation Metrics

Each model is evaluated using:

- Accuracy
- Precision
- Recall
- ROC-AUC Score
- Confusion Matrix

---

## 📈 Visualizations

The project generates the following visualizations:

### ✅ Class Distribution

Displays the balance of shipment outcomes before preprocessing.

### ✅ Correlation Heatmap

Shows relationships between features.

### ✅ Accuracy Comparison

Compares the performance of all models.

### ✅ Precision vs Recall Comparison

Visualizes prediction quality.

### ✅ Confusion Matrix

Displays classification results for each model.

### ✅ ROC Curve Comparison

Compares model performance using ROC-AUC.

---

## 📁 Project Structure

```text
E-Commerce-Shipping-Prediction/
│
├── E-commerce Shipping Dataset.csv
├── main.py
└── README.md
```

---

## 💻 Installation

### Clone the repository

```bash
git clone https://github.com/your-username/E-Commerce-Shipping-Prediction.git
```

### Navigate to the project folder

```bash
cd E-Commerce-Shipping-Prediction
```

### Install the required libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Run the project

```bash
python main.py
```

---

## 📌 Project Features

- Data preprocessing
- Feature encoding
- Correlation analysis
- Multiple machine learning models
- Performance comparison
- Confusion matrix visualization
- ROC curve analysis

---

## Future Improvements

Possible enhancements:

- Hyperparameter tuning
- Cross-validation
- Feature selection
- Random Forest implementation
- XGBoost implementation
- SMOTE for class balancing
- Web application deployment using Flask or Streamlit

---

## 🏆 Conclusion

This project demonstrates how machine learning can be applied to predict e-commerce shipment delivery status and compare different classification algorithms.

The analysis helps identify the strengths and weaknesses of each model and provides insights into factors that influence delivery performance.

---

## Author
https://github.com/Danial-Dirar
https://github.com/Saadman-1
