# Use Python 3.9.6 as the base image
FROM python:3.9.6

# Set the working directory
WORKDIR /app

# Copy the requirements.txt file
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the necessary files
COPY best_NN_model.keras .
COPY best_NN.keras .
COPY decision_tree_scores.pkl .
COPY final_NN.keras .
COPY fitted_svc_model.joblib .
COPY logistic_regression_val_scores.pkl .
COPY m1_workbook.ipynb .
COPY rf_scores.pkl .
COPY svcPred_values .
COPY svm_gamma_scores.pkl .

# Install Jupyter Notebook
RUN pip install jupyter

# Set the entry point to run the Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]