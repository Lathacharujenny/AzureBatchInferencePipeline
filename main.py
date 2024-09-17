# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

from azureml.core import Workspace
import pandas as pd
from azureml.core import Datastore, Dataset
ws = Workspace.from_config()

data = pd.read_csv('data/diabetes2.csv')
data.info()

sample = data[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']].sample(n=100).values
len(sample)

# +
# Create a folder
batch_folder = './batch_data'
os.makedirs(batch_folder, exist_ok=True)

# Save each sample as a separete file
for i in range(100):
    fname = str(i+1)+'.csv'
    sample[i].tofile(os.path.join(batch_folder, fname), sep=',')
# -

# Upload the files to datastore
datastore = ws.get_default_datastore()
datastore.upload(src_dir='batch_data/', target_path='batch_data', overwrite=True, show_progress=True)
batch_data_set = Dataset.File.from_files(path=(datastore, '/batch_data'))
try:
    batch_data_set.register(workspace=ws,
                           name='batch_data',
                           create_new_version=True)
except Exception as ex:
    print(ex)

from azureml.core.compute import ComputeTarget
compute_target = ComputeTarget(workspace=ws, name='AzureInstance')

# +
from azureml.core import Environment
from azureml.core.conda_dependencies import CondaDependencies


env = Environment(name='batch_environment')
dependencies = CondaDependencies.create(conda_packages=['scikit-learn', 'pip'],
                                             pip_packages=['azureml-defaults', 'azureml-core', 'azureml-dataprep[fuse]'])
env.python.conda_dependencies = dependencies
# +
from azureml.pipeline.steps import ParallelRunConfig, ParallelRunStep
from azureml.pipeline.core import PipelineData, Pipeline

output_dir = PipelineData(name='inference', datastore=ws.get_default_datastore(), output_path_on_compute='diabetes/results')
parallel_run_config = ParallelRunConfig(
    source_directory='scripts/',
    entry_script='batch_inference.py',
    environment=env,
    compute_target=compute_target,
    node_count=1,
    output_action='append_row',
    mini_batch_size=5,
    error_threshold=10
)
parallelrun_step = ParallelRunStep(
    name='inference_pipeline',
    parallel_run_config=parallel_run_config,
    inputs=[batch_data_set.as_named_input('diabetes_data')],
    output=output_dir,
    arguments=[],
    allow_reuse=True
)


# +
from azureml.core import Experiment

experiment = Experiment(workspace=ws, name='diabetes_experiment')
pipeline = Pipeline(ws, steps=[parallelrun_step])
pipeline_run = experiment.submit(pipeline)
pipeline_run.wait_for_completion(show_output=True)

# +
import shutil

# Remove the local results folder if left over from a previous run
shutil.rmtree('diabetes-results', ignore_errors=True)

# Get the run for the first step and download its output
prediction_run = next(pipeline_run.get_children())
prediction_output = prediction_run.get_output_data('inference')
prediction_output.download(local_path='diabetes-results')

# +
# Traverse the folder hierarchy and find the results file

for root, dirs, files in os.walk('diabetes-results'):

    for file in files:

        if file.endswith('parallel_run_step.txt'):

            result_file = os.path.join(root,file)

df = pd.read_csv(result_file, delimiter=":", header=None)
df.columns = ["File", "Prediction"]
df.head(20)
# -

published_pipeline = pipeline_run.publish_pipeline(name='diabetes-batch-pipeline', description='Batch scoring of diabetes data', version='1.0')
published_pipeline

print(published_pipeline.endpoint)


