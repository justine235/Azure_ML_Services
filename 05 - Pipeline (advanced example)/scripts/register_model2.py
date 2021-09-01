import argparse
import json
import os
import azureml.core
from azureml.core import Workspace, Experiment, Model
from azureml.core import Run, run
from shutil import copy2
from azureml.core.model import Model
from azureml.pipeline.steps import HyperDriveStepRun

summaryid = [0]
summarymetric = [0]


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="model")
parser.add_argument('--input-dir', dest='input_dir', required=True)
args = parser.parse_args()


run = Run.get_context()
ws = run.experiment.workspace
pipeline_runid = run.get_details()['properties']['azureml.pipelinerunid']
print('pipelineid', pipeline_runid)
#pipeline_run = run.get_run(Experiment(ws, "25082021"),pipeline_runid)
pipeline_run = ws.get_run(pipeline_runid)

for child_run in pipeline_run.get_children():
    listid = child_run.id
    print('id parent', listid)
    for  child_run2 in child_run.get_children():
        print('id pipeline step',child_run2)
        for  child_run3 in child_run2.get_children():
            listid = child_run3.id
            summaryid.append(listid)
            list_metric = child_run3.get_metrics()['Test Precison']
            summarymetric.append(list_metric)

print(summarymetric)
print(summaryid)
maxi = summarymetric.index(max(summarymetric))
maxi_id = summaryid[maxi]
print('best model id : ', maxi_id)

print('start register model')
for child_run3 in child_run2.get_children():
    if child_run3.id == maxi_id:
        child_run3.register_model(model_name='rf', model_path=os.path.join('./outputs/model/saved_model.pkl'))
        metrics = child_run3.get_metrics()
        df = pd.DataFrame(list(metrics.items()),columns = ['metrics','value']) 
        df.to_csv('./test2.csv')
        print("ok it's saved")



run.complete()