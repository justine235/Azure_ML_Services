# get-started/run-hello.py
from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig

ws = Workspace.from_config()
experiment = Experiment(workspace=ws, name='day1-experiment-hello')

config = ScriptRunConfig(source_directory='', script='TEST1-copy.py', compute_target='MLservices')

run = experiment.submit(config)
aml_url = run.get_portal_url()
print(aml_url)