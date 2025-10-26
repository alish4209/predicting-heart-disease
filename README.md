# Coronary Heart Disease Prediction Using Logistic Regression

## Project Overview

This project focuses on predicting the 10-year risk of Coronary Heart Disease (CHD) using patient health data. The model applies Logistic Regression to identify individuals who are at risk of developing CHD within the next decade based on key health indicators such as age, cholesterol level, and blood pressure.

The project demonstrates a complete end-to-end machine learning workflow, including data cleaning, exploratory data analysis (EDA), feature selection, model training, and evaluation.

---

## Dataset Information

The dataset used in this project is the **Framingham Heart Study dataset**. It contains medical and demographic information for several patients, including whether they developed coronary heart disease within ten years.

**Key features used for prediction:**

* age: Age of the patient in years
* Sex_male: 1 if the patient is male, 0 if female
* cigsPerDay: Average number of cigarettes smoked per day
* totChol: Total cholesterol level
* sysBP: Systolic blood pressure
* glucose: Blood glucose level

**Target variable:**

* TenYearCHD: Indicates whether the patient developed CHD within 10 years (1 = Yes, 0 = No)

## Project Workflow

1. **Importing Libraries**
   Essential Python libraries like pandas, numpy, matplotlib, seaborn, and scikit-learn are imported for data handling,  visualization, and model building.

2. **Data Preparation**

   * Loaded the dataset using pandas.
   * Dropped irrelevant or less significant columns such as 'education'.
   * Renamed columns for clarity (e.g., 'male' to 'Sex_male').
   * Removed missing values to ensure data quality.

3. **Exploratory Data Analysis (EDA)**

   exploratory data analysis to understand how many people were at risk versus not at risk.
   Visualized the distribution of the target variable (`TenYearCHD`) using seaborn count plots.
   Checked class imbalance and feature correlations to understand data patterns.

5. **Feature Selection**

   * Selected key medical and lifestyle variables that are most relevant to CHD risk.

6. **Data Scaling**

   * Standardized the features using StandardScaler to bring all variables to a similar range (mean = 0, standard deviation = 1).

7. **Train-Test Split**

   * Divided the dataset into training and testing sets (70% training, 30% testing) to evaluate model performance on unseen data.

8. **Model Building**

   * Built a Logistic Regression model using scikit-learn’s `LogisticRegression()` function.
   * Trained the model on the training dataset using the `.fit()` method.
   * Predicted the outcomes on the test dataset using `.predict()`.

9. **Model Evaluation**

   * Calculated Accuracy, Precision, and Recall to measure performance.
   * Generated a Confusion Matrix and visualized it using seaborn’s heatmap to interpret results.

---

## Evaluation Metrics

* **Accuracy:** Percentage of overall correct predictions.
* **Precision:** Proportion of predicted positive cases that were actually positive.
* **Recall:** Proportion of actual positive cases correctly identified by the model.

These metrics provide a balanced understanding of how well the model identifies CHD cases while minimizing false predictions.

---

## Results Summary

The Logistic Regression model achieved a reasonable level of accuracy and provided meaningful insights into the factors influencing coronary heart disease. The evaluation metrics show that the model can effectively predict high-risk individuals, making it a valuable step toward preventive healthcare analytics.

---

## Tools and Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

## Conclusion

This project demonstrates how Logistic Regression can be applied to real-world healthcare data for risk prediction. The model not only helps in understanding critical health indicators but also emphasizes the importance of data preprocessing, model evaluation, and interpretability in machine learning workflows.

Author :- Alish
