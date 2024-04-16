import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
import ipdb
import logging
import warnings
import mlflow
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

train = pd.read_csv('data/train.csv')
scaler = StandardScaler()

estimators = 80
depth = 13



exp = mlflow.set_experiment(experiment_name="Hotels_random_forest")

with mlflow.start_run(experiment_id=exp.experiment_id):

    #train model

    train_scaled = pd.DataFrame(scaler.fit_transform(train.drop(columns=['booking_status'])), 
                            columns=train.drop(columns=['booking_status']).columns)

    train_scaled['booking_status'] = train['booking_status']

    X = train_scaled.drop(columns=['booking_status'])
    y = train_scaled['booking_status']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    forest_model = RandomForestClassifier(n_estimators=estimators, max_depth=depth)
    forest_model.fit(X_train, y_train)
    y_pred = forest_model.predict(X_test)

    # log important stuff about model 

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred)
    }

    params = {
        "estimators": estimators,
        "depth": depth
    }
    
    tags = {
        "squad": "MarketOps",
        "release.version": "1.0"
    }
    
    signature = infer_signature(X_test, y_pred)
    input_example = {
        "columns":np.array(X_test.columns),
        "data": np.array(X_test.values)
    }

    mlflow.set_tags(tags)
    mlflow.log_metrics(metrics)
    mlflow.log_params(params)
    mlflow.log_artifacts("data/")
    mlflow.sklearn.log_model(forest_model, "forest_model", signature=signature, input_example=input_example)

    


run = mlflow.last_active_run()

print("---------------------------------------------------------")
print(f"Experiment Name: {exp.name}")
print(f"Experiment ID: {exp.experiment_id}")
print(f"Run Name: {run.info.run_name}")
print(f"Run ID: {run.info.run_id}")
print("---------------------------------------------------------")

ipdb.set_trace()