**Loan Prediction Project - README**

### Project Overview
This project is a machine learning model to predict loan approval status based on applicant details. The model is built using Python and Jupyter Notebook.

### Files Included
- `LOAN PREDICTION.ipynb`: Jupyter Notebook containing data analysis, model building, and evaluation.
- `requirements.txt`: List of required Python packages (e.g., pandas, numpy, scikit-learn, matplotlib, seaborn).
- `README.md`: Project documentation (this file).

### Prerequisites
- Python 3.9 or above
- Jupyter Notebook
- Install required packages using:
  ```bash
  pip install -r requirements.txt
  ```

### Dataset
- The dataset contains information about loan applicants including gender, marital status, income, loan amount, credit history, and loan status.

### Steps Implemented in the Notebook
1. **Data Loading:** Load dataset using Pandas.
2. **Exploratory Data Analysis (EDA):** Analyze data distribution and identify patterns using Matplotlib and Seaborn.
3. **Data Preprocessing:** Handle missing values, encode categorical variables, and scale numerical features.
4. **Model Building:** Train a machine learning model (e.g., Logistic Regression) using scikit-learn.
5. **Model Evaluation:** Evaluate model performance using metrics such as accuracy, precision, recall, and confusion matrix.
6. **Results and Insights:** Present key findings and insights from the model's performance.

### How to Run the Project
1. Clone the repository or download the project files.
2. Install dependencies with `pip install -r requirements.txt`.
3. Open `LOAN PREDICTION.ipynb` in Jupyter Notebook.
4. Run all cells to execute the project workflow.

### Results
- The model achieved an accuracy score of accuracy_score.
- Important features influencing loan approval were NUMBER_OF_INSTALLMENTS, SANCTION_AMT, OVER_DUE_AMT, INSTALMENT_LOAN_TYPE_ConsumerLoan,INSTALMENT_LOAN_TYPE_OtherInstalmentOperation, loan_status_Existing
  

### Future Improvements
- Experiment with different models (e.g., Random Forest, XGBoost).
- Perform hyperparameter tuning.
- Enhance data preprocessing techniques.

### Author
- Developed by: Shalini

