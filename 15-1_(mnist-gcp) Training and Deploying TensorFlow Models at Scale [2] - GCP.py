# Python ≥3.5 is required
import sys

assert sys.version_info >= (3, 5)
# Scikit-Learn ≥0.20 is required
import sklearn

assert sklearn.__version__ >= "0.20"
# TensorFlow ≥2.0 is required
import tensorflow as tf
from tensorflow import keras

assert tf.__version__ >= "2.0"
# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
np.random.seed(42)
tf.random.set_seed(42)
# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "deploy"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train_full = X_train_full[..., np.newaxis].astype(np.float32) / 255.
X_test = X_test[..., np.newaxis].astype(np.float32) / 255.
X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
print("\n\n###############################################################")
print("And now let's use saved_model_cli to make predictions \n"
      "for the instances we just saved:")
print("let's start by creating the query.")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "hknu-ai-a29fe866f93c.json"
# pip install google-api-python-client==1.7.9
import googleapiclient.discovery
from google.api_core.client_options import ClientOptions
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "project15-279813-3066b58c66e5.json"
project_id = "project15-279813"
model_id = "my_mnist_model"
model_path = "projects/{}/models/{}".format(project_id, model_id)
model_path += "/versions/v0001/"  # if you want to run a specific version
ml_resource = googleapiclient.discovery.build("ml", "v1").projects()
print("\nmodel_path: \n", model_path)
# function use to request a prediction from the web api of the model and get a reponse of the predctions


def predict(X):
    input_data_json = {"signature_name": "serving_default",
                       "instances": X.tolist()}
    request = ml_resource.predict(name=model_path, body=input_data_json)
    response = request.execute()
    print("\nresponse: \n", response)
    if "error" in response:
        raise RuntimeError(response["error"])
    return np.array([pred['dense_1'] for pred in response["predictions"]])
    # response 를 프린트해 보면 pred['****'] 안에 들어 갈 단어('dense_4')를 알 수 있음


X_new = X_test[:10]
# predict the results for the test set
print("\n\npredict(X_test[:10]): \n",
      predict(X_new))
