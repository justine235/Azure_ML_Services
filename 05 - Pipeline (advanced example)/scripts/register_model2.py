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
        child_run3.register_model(model_name='rf', model_path='./outputs/model/saved_model.pkl') 
        print("ok it's saved")


# get each pipeline step ID
#for child_run in run.get_children():
#    listid = child_run.id
 #   print(listid)

#for child_run in run.get_children():
#    listid = child_run.id
#    print(listid)
#    summaryid.append(listid)
#    list_metric = child_run.get_metrics()['Test Precison']
#    print(list_metric)
#    summarymetric.append(list_metric)

#pipeline_run = run.get_run('Experiment(ws, '25082021')',pipeline_runid)
#pipeline_run = run.get_run(Experiment(ws, '25082021'),pipeline_runid,rehydrate=True)

#step_run = HyperDriveStepRun(step_run = pipeline_run.find_step_run('hd_step')[0])
#best_run = step_run.get_best_run_by_primary_metric()
#metrics = best_run.get_metrics()
#details = best_run.get_details()


#best_run.register_model(model_name='model_rf',
#                   description = 'best one',
#                   model_path='./outputs',
#                   tags={'Run_ID',details['runID']
#                   })

# Just for the best run
#print("BEST RUN:", best_run_id)
#print("Hyperparameters for best run:\n", hyperparameters[best_run_id])
#print("Metrics of best run:\n", metrics[best_run_id])

run.complete()