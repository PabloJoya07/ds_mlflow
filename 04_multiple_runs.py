import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import ipdb
import logging
import warnings
import mlflow

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

train = pd.read_csv('data/train.csv')
scaler = StandardScaler()

estimators = list(range(150, 50, -10))
depth = list(range(5, 15))

ipdb.set_trace()
exp = mlflow.set_experiment(experiment_name="Hotels_multiple_execution")

for est,dep in zip(estimators, depth):
    with mlflow.start_run(experiment_id=exp.experiment_id):

        #train model
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

        forest_model = RandomForestClassifier(n_estimators=est, max_depth=dep)
        forest_model.fit(X_train, y_train)
        y_pred = forest_model.predict(X_test)

    run = mlflow.last_active_run()

    print("---------------------------------------------------------")
    print(f"Experiment Name: {exp.name}")
    print(f"Experiment ID: {exp.experiment_id}")
    print(f"Run Name: {run.info.run_name}")
    print(f"Run ID: {run.info.run_id}")
    print("---------------------------------------------------------")