import argparse
import json
import os
import azureml.core
from azureml.core import Workspace, Experiment, Model
from azureml.core import Run
from shutil import copy2
from azureml.core.model import Model

parser = argparse.ArgumentParser()
parser.add_argument('--saved-model', type=str, dest='saved_model', help='path to saved model file')
args = parser.parse_args()
run = Run.get_context()
run.log("args.saved_model",args.saved_model)


ws = Run.get_context().experiment.workspace
model = Model.register(workspace=ws, model_name='rf', model_path=args.saved_model)

#model = run.register_model(model_name='keras-mlp-mnist',
#                           model_path=model_path,
#                           datasets =[('training data',train_dataset)])

run.complete()