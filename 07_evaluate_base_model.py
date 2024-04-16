import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
import ipdb
import logging
import warnings
import mlflow

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

train = pd.read_csv('data/train.csv')
scaler = StandardScaler()

exp = mlflow.set_experiment(experiment_name="Hotels_Best_Models")

with mlflow.start_run(experiment_id=exp.experiment_id):

    mlflow.sklearn.autolog(
        log_input_examples=False,
        log_model_signatures=True,
        log_models=True,
        log_datasets=False
    )

    train_scaled = pd.DataFrame(scaler.fit_transform(train.drop(columns=['booking_status'])), 
                                columns=train.drop(columns=['booking_status']).columns)

    train_scaled['booking_status'] = train['booking_status']

    X = train_scaled.drop(columns=['booking_status'])
    y = train_scaled['booking_status']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    dummy_model = DummyClassifier(strategy="most_frequent")
    dummy_model.fit(X_train, y_train)
    y_pred = dummy_model.predict(X_test)

    mlflow.evaluate(
        mlflow.get_artifact_uri("model"),
        train_scaled,
        targets="booking_status",
        model_type="classifier"
    )

run = mlflow.last_active_run()

print("---------------------------------------------------------")
print(f"Experiment Name: {exp.name}")
print(f"Experiment ID: {exp.experiment_id}")
print(f"Run Name: {run.info.run_name}")
print(f"Run ID: {run.info.run_id}")
print("---------------------------------------------------------")
