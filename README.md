# CODE-ALPHA-INTERNSHIP
## Task 1: Credit Scoring Model

**Objective:** Predict a person's creditworthiness (the likelihood of default) from past financial data.

**Dataset:** "Default of Credit Card Clients" (UCI Machine Learning Repository / Kaggle). 30,000 credit card clients from Taiwan, with 23 features (credit limit, demographics, 6 months of repayment status, bill amounts and payment amounts). Target: `default.payment.next.month` (about 22% defaulters).

**Approach:**
- Data exploration and cleaning (undocumented `EDUCATION` and `MARRIAGE` values merged into "others", `ID` dropped)
- Feature engineering: 9 new features such as credit utilization, payment ratio, number of late months, maximum delay and bill trend
- Stratified 80/20 train/test split and standard scaling
- Models: Logistic Regression, Decision Tree and Random Forest (all with balanced class weights)
- Evaluation: Precision, Recall, F1-Score and ROC-AUC

**Results (test set, Default class):**

| Model | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.4355 | 0.6059 | 0.5068 | 0.7493 |
| Decision Tree | 0.4286 | 0.6473 | 0.5158 | 0.7621 |
| Random Forest | 0.5043 | 0.5750 | 0.5373 | 0.7765 |

**Key findings:** Random Forest performed best overall. Payment delay features (number of late months, maximum delay) showed the strongest relationship with default.

**Tools:** Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, Google Colab

## Task 3: Handwritten Character Recognition

Recognizing handwritten digits (0 to 9) using a Convolutional Neural Network (CNN) on the MNIST dataset.

- **Dataset:** MNIST (60,000 train + 10,000 test images, 28x28 grayscale), loaded from `keras.datasets.mnist`
- **Model:** CNN with 2 Conv + 2 MaxPooling layers, Dropout (0.5) and 2 Dense layers (225,034 parameters)
- **Training:** Adam optimizer, categorical crossentropy, 10 epochs, batch size 128
- **Result:** **99.24% test accuracy**, test loss 0.0223 (76 of 10,000 images misclassified)


  ## Task 4: Diabetes Prediction using Random Forest

**CodeAlpha Machine Learning Internship | Task 4: Disease Prediction from Medical Data**

## Objective
To predict whether a patient is likely to have diabetes based on medical measurements, using a Random Forest classifier.

## Dataset
Pima Indians Diabetes Dataset (UCI ML Repository)
- 768 patient records, 8 input features, 1 target column
- **Features:** Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
- **Target:** Outcome (1 = Diabetic, 0 = Non-diabetic)
- Class distribution: 500 non-diabetic, 268 diabetic

## Approach
1. Loaded and explored the data
2. Cleaned the data: zero values in Glucose, BloodPressure, SkinThickness, Insulin and BMI are not valid, so they were treated as missing values
3. Exploratory data analysis (class balance, distributions, correlation heatmap)
4. Stratified 80/20 train-test split
5. Filled missing values with the median of the training set (after the split, to avoid data leakage)
6. Trained a baseline Random Forest model
7. Tuned hyperparameters with GridSearchCV (5-fold cross-validation, ROC-AUC scoring)
8. Evaluated both models and analyzed feature importance
9. Saved the final model with joblib

## Results

| Metric | Baseline | Tuned |
|---|---|---|
| Accuracy | 0.740 | 0.727 |
| Precision | 0.652 | 0.588 |
| Recall | 0.556 | 0.741 |
| F1-Score | 0.600 | 0.656 |
| ROC-AUC | 0.812 | 0.821 |

Best parameters: `max_depth=5`, `min_samples_leaf=4`, `min_samples_split=2`, `n_estimators=300`

The tuned model detects about 74% of diabetic patients (recall), compared to 56% for the baseline. In medical screening, missing a diabetic patient is more harmful than a false alarm, so the tuned model is the better choice.

## Key Findings
- **Glucose** is the most important feature (0.355), followed by **BMI** (0.179) and **Age** (0.128).
- Blood glucose level, body weight and age are the main factors the model uses to predict diabetes.

## Tools and Libraries
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Joblib, Google Colab

## Repository Structure
```
CodeAlpha_Diabetes_Prediction/
├── diabetes_prediction.ipynb    # Main notebook
├── diabetes_dataset.csv         # Dataset
├── diabetes_rf_model.pkl        # Saved model
└── README.md
```

## How to Run
1. Open the notebook in Google Colab or Jupyter Notebook
2. Upload `diabetes_dataset.csv`
3. Run all cells in order

## Limitations and Future Work
- The dataset is small (768 records) and moderately imbalanced.
- Other models such as XGBoost, SVM and Logistic Regression can be compared.
- Techniques like SMOTE or threshold tuning can improve precision and recall further.
- This project is for learning purposes and is not a substitute for medical diagnosis.

