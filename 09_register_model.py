import mlflow
import ipdb

import mlflow

# Obtén todos los modelos registrados
#registered_models = mlflow.MlflowClient().get_registered_model("Tourism_model")
#run_id = registered_models.latest_versions[0].run_id
#run_obj = mlflow.get_run(run_id)
#accuracy = run_obj.data.metrics.accuracy_score

ipdb.set_trace()

criteria_filter = "name = 'Hotels_Best_Models'"
experiments = mlflow.search_experiments(filter_string=criteria_filter)
exp_id = experiments[0].experiment_id

df = mlflow.search_runs([exp_id], order_by=["metrics.accuracy_score DESC"])
run_obj = mlflow.get_run(df.at[0, 'run_id'])

ipdb.set_trace()
client = mlflow.MlflowClient()
client.create_registered_model("Tourism_model_stg")

result = client.create_model_version(
    name="Tourism_model_stg",
    source=run_obj.info.artifact_uri,
    run_id=run_obj.info.run_id,
)