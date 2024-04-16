import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from mlflow.models import MetricThreshold
import ipdb
import logging
import warnings
import mlflow

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

train = pd.read_csv('data/train.csv')
scaler = StandardScaler()

estimators = 300
depth = 20

criteria_filter = "name = 'Hotels_Best_Models'"
experiments = mlflow.search_experiments(filter_string=criteria_filter)
exp_id = experiments[0].experiment_id

df = mlflow.search_runs([exp_id], order_by=["metrics.accuracy_score DESC"])
run_obj = mlflow.get_run(df.at[0, 'run_id'])
uri_baseline = f'{run_obj.info.artifact_uri}/model'

thresholds = {
    "accuracy_score": MetricThreshold(
        threshold=float(df.at[0, 'metrics.accuracy_score']),
        min_absolute_change=0.1, # accuracy should be at least 0.05 greater than baseline model accuracy
        min_relative_change=0.01,# accuracy should be at least 5 percent greater than baseline model accuracy
        greater_is_better=True
    )
}

exp = mlflow.set_experiment(experiment_name="Hotels_Best_Models")

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

    forest_model = RandomForestClassifier(n_estimators=estimators, max_depth=depth)
    forest_model.fit(X_train, y_train)
    y_pred = forest_model.predict(X_test)

    mlflow.evaluate(
        mlflow.get_artifact_uri("model"),
        train_scaled,
        targets="booking_status",
        model_type="classifier",
        validation_thresholds=thresholds,
        baseline_model=uri_baseline,
    )
    

run = mlflow.last_active_run()

print("---------------------------------------------------------")
print(f"Experiment Name: {exp.name}")
print(f"Experiment ID: {exp.experiment_id}")
print(f"Run Name: {run.info.run_name}")
print(f"Run ID: {run.info.run_id}")
print("---------------------------------------------------------")

ipdb.set_trace()