from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-GUI environments
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score as ras, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
import os

app = Flask(__name__, static_url_path='/static')

# Function to correct data types
def correct_dtypes(df):
    for col in df.columns:
        try:
            if 'date' in col.lower() or 'time' in col.lower():
                df[col] = pd.to_datetime(df[col], errors='coerce')
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        except Exception as e:
            print(f"Could not convert column {col}: {e}")
    return df

chunk_size = 100000
chunks = []
for chunk in pd.read_csv('PS_20174392719_1491204439457_log.csv', chunksize=chunk_size, low_memory=False):
    chunk = correct_dtypes(chunk)
    chunks.append(chunk)

data = pd.concat(chunks, ignore_index=True)

fraud = data[data.isFraud == 1]
not_fraud = data[data.isFraud == 0]

not_fraud_downsampled = resample(not_fraud, 
                                 replace=False, 
                                 n_samples=len(fraud), 
                                 random_state=42)

data_downsampled = pd.concat([fraud, not_fraud_downsampled])

type_new = pd.get_dummies(data_downsampled['type'], drop_first=True)
data_new = pd.concat([data_downsampled, type_new], axis=1)
X = data_new.drop(['isFraud', 'type', 'nameOrig', 'nameDest'], axis=1)
y = data_new['isFraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(max_iter=10000, solver='lbfgs'),
    "XGBoost": XGBClassifier(enable_categorical=False, verbosity=1, eval_metric='logloss', tree_method='hist'),
    "SVC": SVC(kernel='rbf', probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=7)
}

@app.route('/')
def index():
    # Select a subset of the data to display (e.g., the first 10 rows)
    table_data = data_downsampled.head(10).to_dict(orient='records')
    columns = data_downsampled.columns.tolist()
    return render_template('index.html', table_data=table_data, columns=columns)

@app.route('/train', methods=['POST'])
def train_models():
    results = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        
        train_preds = model.predict_proba(X_train)[:, 1]
        train_roc_auc = ras(y_train, train_preds)
        
        y_preds = model.predict_proba(X_test)[:, 1]
        val_roc_auc = ras(y_test, y_preds)
        
        results[model_name] = {
            "train_roc_auc": train_roc_auc,
            "val_roc_auc": val_roc_auc
        }

    # Generate and save visualizations after training
    hist_plot, crm_plot = generate_visualizations()

    return render_template('index.html', results=results, hist_plot=hist_plot, crm_plot=crm_plot)

@app.route('/confusion_matrix')
def plot_confusion_matrix():
    xgb_model = models["XGBoost"]
    y_pred_class = xgb_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_class)

    # Generate and save confusion matrix plot
    cm_path, heatmap_path = generate_confusion_matrix_plots(cm)

    return render_template('index.html', cm_path=cm_path, heatmap_path=heatmap_path)

@app.route('/visualization')
def generate_visualizations():
    # Print statements for debugging
    print("data_downsampled.head():")
    print(data_downsampled.head())
    print("data_downsampled['type'].value_counts():")
    print(data_downsampled['type'].value_counts())
    print("data_downsampled['amount'].describe():")
    print(data_downsampled['amount'].describe())
    
    # Histogram Plot
    plt.figure(figsize=(10, 5))
    sns.histplot(data_downsampled['step'], bins=50)
    plt.title('Distribution of step variable')
    hist_plot = os.path.join('static', 'histplot.png')
    plt.savefig(hist_plot)
    plt.close()
    
    # Select only numeric columns
    numeric_data = data_downsampled.select_dtypes(include=[np.number])

    # List of columns to exclude from the correlation matrix
    columns_to_exclude = ['nameOrig', 'nameDest', 'type', 'isFlaggedFraud']

    # Dynamically filter out the unwanted columns
    filtered_numeric_data = numeric_data.drop(columns=columns_to_exclude, errors='ignore')

    # Plot correlation matrix
    plt.figure(figsize=(10, 12))
    sns.heatmap(filtered_numeric_data.corr(), cmap='BrBG', annot=True, fmt='.2f', linewidths=2)
    plt.title('Correlation matrix')
    crm_plot = os.path.join('static', 'correlation.png')
    plt.savefig(crm_plot)
    plt.close()

    return hist_plot, crm_plot

def generate_confusion_matrix_plots(cm):
    # Confusion Matrix Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix for XGBoost Model')
    cm_path = os.path.join('static', 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()

    # Heatmap Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Heatmap for Confusion Matrix')
    heatmap_path = os.path.join('static', 'heatmap.png')
    plt.savefig(heatmap_path)
    plt.close()

    return cm_path, heatmap_path

if __name__ == '__main__':
    app.run(debug=True)
