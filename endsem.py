# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause


# PART: library dependencies -- sklear, torch, tensorflow, numpy, transformers

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
import pdb
import argparse
from sklearn.metrics import f1_score
from utils import (
    preprocess_digits,
    train_dev_test_split,
    h_param_tuning,
    data_viz,
    pred_image_viz,
    get_all_h_param_comb,
    tune_and_save,
)
from joblib import dump, load

parser = argparse.ArgumentParser(
                    prog = 'ProgramName',
                    description = 'What the program does',
                    epilog = 'Text at the bottom of help')

        # positional argument
parser.add_argument('--clf_name') 
parser.add_argument('--random')  

args = parser.parse_args()


train_frac, dev_frac, test_frac = 0.8, 0.1, 0.1
assert train_frac + dev_frac + test_frac == 1.0

# 1. set the ranges of hyper parameters
gamma_list = [0.02]
c_list = [0.2]

params = {}
params["gamma"] = gamma_list
params["C"] = c_list

h_param_comb = get_all_h_param_comb(params)


# PART: load dataset -- data from csv, tsv, jsonl, pickle
digits = datasets.load_digits()
data_viz(digits)
data, label = preprocess_digits(digits)
# housekeeping
del digits


x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(
    data, label, train_frac, dev_frac,random= int(args.random),shuffle=True
)

# PART: Define the model
# Create a classifier: a support vector classifier

if args.clf_name=="svm":
    clf = svm.SVC()
# define the evaluation metric
metric = metrics.accuracy_score


actual_model_path = tune_and_save(
    clf, x_train, y_train, x_dev, y_dev, metric, h_param_comb, model_path="./models/svmjoblib.joblib"
)


# 2. load the best_model
best_model = load(actual_model_path)

# PART: Get test set predictions
# Predict the value of the digit on the test subset
predicted = best_model.predict(x_test)

pred_image_viz(x_test, predicted)

# 4. report the test set accurancy with that best model.
# PART: Compute evaluation metrics
resultsFile=open('results/svm_'+args.clf_name+args.random+'.txt','a')

resultsFile.write(   f"Test accuracy {clf}:\n")
resultsFile.write( f"{metrics.classification_report(y_test, predicted)}\n")
resultsFile.write(   f"F1 score {clf}:\n")
f1=f1_score(y_test, predicted, pos_label=1, average='macro', zero_division='warn')
resultsFile.write(str(f1))
resultsFile.write('model saved at ./models/svmjoblib.joblib')

print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)
