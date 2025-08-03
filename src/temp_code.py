from mlflow.tracking import MlflowClient

client = MlflowClient()

# View ALL experiments (active + deleted)
experiments = client.search_experiments(view_type="ALL")

for exp in experiments:
    print(f"Name: {exp.name}, ID: {exp.experiment_id}, Status: {exp.lifecycle_stage}")
