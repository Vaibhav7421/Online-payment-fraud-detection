import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score as ras, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
import dask.dataframe as dd
import os


# Function to correct data types
def correct_dtypes(df):
    for col in df.columns:
        if col in datetime_columns:
            try:
                df[col] = pd.to_datetime(df[col], format=datetime_columns[col])
            except ValueError:
                df[col] = pd.to_datetime(df[col], errors='coerce') # handle invalid dates
        else:
            try:
                df[col] = pd.to_numeric(df[col])
            except ValueError:
                pass
    return df
# Specify datetime columns with their formats
datetime_columns = {
    # 'column_name': 'format_string' # add your datetime columns and formats here
    # Example: 'date_column': '%Y-%m-%d'
}

# Read the CSV file with pandas
file_path = 'PS_20174392719_1491204439457_log.csv'
data = pd.read_csv(file_path, dtype=str, low_memory=False)

# Correct data types dynamically
data = correct_dtypes(data)

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Display information about the dataset
print("\nInformation about the dataset:")
print(data.info())

# Convert the pandas DataFrame to a Dask DataFrame
ddf = dd.from_pandas(data, npartitions=4)

# Downsample the majority class (Not Fraud)
fraud = ddf[ddf.isFraud == 1].compute()
not_fraud = ddf[ddf.isFraud == 0].compute()

not_fraud_downsampled = resample(not_fraud, 
                                 replace=False, 
                                 n_samples=len(fraud), 
                                 random_state=42)

data_downsampled = pd.concat([fraud, not_fraud_downsampled])
# Check data_downsampled before plotting
print("data_downsampled.head():")
print(data_downsampled.head())
print("data_downsampled['type'].value_counts():")
print(data_downsampled['type'].value_counts())
print("data_downsampled['amount'].describe():")
print(data_downsampled['amount'].describe())

plt.figure(figsize=(10, 5))
sns.countplot(x='type', data=data)
plt.title('Count of transactions by type')
plt.show()
countplot=os.path.join('static','countplot.png')
plt.savefig(countplot)

plt.figure(figsize=(10, 5))
sns.barplot(x='type', y='amount', data=data)
plt.title('Average transaction amount by type')
plt.show()
barplot=os.path.join('static','barplot.png')
plt.savefig()
print("Distribution of 'isFraud' variable:")
print(data_downsampled['isFraud'].value_counts())

plt.figure(figsize=(10, 5))
sns.histplot(data_downsampled['type'], bins=50)
plt.title('Distribution of step variable')
plt.show()
# Select only numeric columns
numeric_data = data_downsampled.select_dtypes(include=[np.number])

# List of columns to exclude from the correlation matrix
columns_to_exclude = ['nameOrig', 'nameDest', 'type']  

# Dynamically filter out the unwanted columns
filtered_numeric_data = numeric_data.drop(columns=columns_to_exclude, errors='ignore')

# Plot correlation matrix
plt.figure(figsize=(12, 6))
sns.heatmap(filtered_numeric_data.corr(), cmap='BrBG', annot=True, fmt='.2f', linewidths=2)
plt.title('Correlation matrix')
plt.show()

# Data preprocessing
type_new = pd.get_dummies(data_downsampled['type'], drop_first=True)
data_new = pd.concat([data_downsampled, type_new], axis=1)
X = data_new.drop(['isFraud', 'type', 'nameOrig', 'nameDest'], axis=1)
y = data_new['isFraud']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Training and evaluation
models = [
    LogisticRegression(max_iter=10000, solver='lbfgs'), 
    XGBClassifier(verbosity=1, tree_method='hist', eval_metric='logloss'), 
    SVC(kernel='rbf', probability=True), 
    RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=7)
]

for model in models:
    model.fit(X_train, y_train)
    print(f'{model.__class__.__name__} : ')
    
    train_preds = model.predict_proba(X_train)[:, 1]
    print('Training ROC AUC Score: ', ras(y_train, train_preds))
    
    y_preds = model.predict_proba(X_test)[:, 1]
    print('Validation ROC AUC Score: ', ras(y_test, y_preds))
    print()

# Plot confusion matrix 
xgb_model = models[1]
y_pred_class = xgb_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_class)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for XGBoost Model')
plt.show()


