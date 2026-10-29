# Credit Card Customer Classification Using Machine Learning

## Project Overview

This project uses Machine Learning to classify credit card customers as **Good Customers** or **Bad Customers** based on their financial and demographic information.

The project includes data preprocessing, missing-value handling, feature scaling, model training, and a Flask web application that accepts user inputs and displays the prediction.

## Features

- Customer classification using Machine Learning.
- Missing-value handling using Random Sample Imputation.
- Feature scaling using `StandardScaler`.
- Multiple classification algorithms for model evaluation.
- Interactive web interface built with Flask.
- Logging for tracking model evaluation results and errors.

## Technologies Used

- **Programming Language:** Python
- **Libraries:** NumPy, Pandas, Scikit-learn
- **Machine Learning Algorithms:** K-Nearest Neighbors (KNN), Naive Bayes, Logistic Regression, Decision Tree, Random Forest, AdaBoost, Gradient Boosting, XGBoost
- **Data Visualization:** Matplotlib, Seaborn
- **Web Framework:** Flask
- **Model Serialization:** Pickle
- **Data Processing:** Random Sample Imputation and StandardScaler

## Project Structure

```text
Credit-Card-Customer-Classification/
│
├── app.py
├── all_models.py
├── random_sample_technique.py
├── log_code.py
├── Credit_card_Model.pkl
├── scaled_model.pkl
├── templates/
│   └── index.html
├── logs/
└── README.md
```

*Note: Update the filenames in this structure to match your actual GitHub repository.*

## Machine Learning Workflow

1. **Data Preparation:** Load and prepare customer data for analysis.
2. **Missing-Value Handling:** Apply Random Sample Imputation to handle missing values.
3. **Model Training:** Train and evaluate multiple classification algorithms.
4. **Feature Scaling:** Apply the saved scaler to transform input features before prediction.
5. **Model Prediction:** Use the trained Gradient Boosting model to classify customers.
6. **Web Application:** Use Flask to collect user inputs and display the prediction.

## Input Features

The application uses the following 12 features:

| Feature | Description |
|---|---|
| Age | Customer's age |
| NumberOfOpenCreditLinesAndLoans | Number of open credit lines and loans |
| NumberRealEstateLoansOrLines | Number of real-estate loans or credit lines |
| MonthlyIncome_replaced | Monthly income after preprocessing |
| Gender_Male | Encoded gender feature |
| Region_East | Encoded East region |
| Region_North | Encoded North region |
| Region_South | Encoded South region |
| Region_West | Encoded West region |
| Rented_OwnHouse_con | Encoded housing information |
| Occupation_con | Encoded occupation information |
| Education_con | Encoded education information |

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
```

Replace the URL and repository name with your actual GitHub details.

### 2. Install Dependencies

```bash
pip install flask numpy pandas scikit-learn matplotlib seaborn xgboost
```

### 3. Run the Application

Ensure that the trained model files, scaler file, and `templates/index.html` are available in the correct locations.

```bash
python app.py
```

### 4. Open the Application

Open your browser and visit:

```text
http://127.0.0.1:5000
```

Enter the required customer information and submit the form to view the classification result.

## Prediction Results

The application displays one of the following results:

- **Good Customer:** The model predicts that the customer belongs to class `1`.
- **Bad Customer:** The model predicts that the customer belongs to class `0`.

These are model predictions and should not be treated as definitive assessments of a customer's creditworthiness.

## Model Evaluation

The project evaluates classification models using:

- Accuracy Score
- Confusion Matrix
- Classification Report
- ROC Curve

The evaluation results are recorded using the project's logging setup.

## Future Improvements

- Improve model performance through hyperparameter tuning.
- Add model comparison charts and evaluation metrics.
- Improve input validation and error handling.
- Deploy the Flask application to a cloud platform.
- Enhance the web interface for a better user experience.

## Author

**Pamarthi Devanadh**

Data Science Intern at Vihara Tech Institute

**Areas of Interest:** Data Science, Machine Learning, Artificial Intelligence

---

If you find this project useful, feel free to star the repository!
