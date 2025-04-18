

---

#  Heart Disease Prediction Using Machine Learning

This project aims to predict the presence of heart disease in patients using supervised machine learning algorithms. It was developed as the final project for my Machine Learning course.

## Overview

The project applies various classification algorithms, performs hyperparameter tuning, handles class imbalance using SMOTE, and evaluates model performance through metrics and visualizations.

##  Objectives

- Train multiple classification models
- Handle class imbalance using SMOTE
- Tune hyperparameters with GridSearchCV
- Evaluate models using Accuracy, Precision, Recall, F1-Score, ROC AUC, and PR AUC
- Compare model performances visually
- Identify important features for prediction

## Tools & Technologies

- Python
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn
- SMOTE (from imbalanced-learn)
- Google Colab

##  Models Used

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- Neural Network (MLPClassifier)
- Voting Classifier (Ensemble Model)

##  Performance Evaluation

- Confusion Matrix
- Classification Report
- ROC Curve
- Precision-Recall AUC
- Feature Importance (Random Forest)
- Performance Comparison Table

##  Project Structure

```
├── data/
│   └── heart.csv
├── notebook/
│   └── heart_disease_prediction.ipynb
├── model_performance_comparison.csv
├── README.md
```

## ⚙️ Key Techniques

- **SMOTE**: Used to balance the dataset due to the minority class (heart disease cases).
- **Scaling**: StandardScaler applied for models sensitive to feature scaling (e.g., SVM, Neural Network).
- **Hyperparameter Tuning**: GridSearchCV used for model optimization.

##  Results

- Ensemble model (Voting Classifier) achieved the best performance.
- Models were evaluated based on F1 Score and ROC AUC.
- Feature importance was visualized to understand influential variables.




---

