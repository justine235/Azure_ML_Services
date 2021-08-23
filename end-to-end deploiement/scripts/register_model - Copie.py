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
parser.add_argument('--model-file', type=str, dest='model_file', help='path to metrics data file')
parser.add_argument('--model-description', type=str, dest='model_desc', help='description')
parser.add_argument('--model-name', type=str, dest='model_name', help='model name')

args = parser.parse_args()
run = Run.get_context()
ws = Run.get_context().experiment.workspace

model_output_dir = "./model/" + args.model_file
os.makedirs(model_output_dir, exist_ok=True)
run.upload_file(model_output_dir, args.saved_model)

#model = Model.register(workspace=ws, model_name='rf', model_path=args.saved_model)

run.register_model(model_name=args.model_name,
                   description = args.model_desc,
                   model_path=model_output_dir)

run.complete()