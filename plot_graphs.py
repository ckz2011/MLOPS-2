# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause


# PART: library dependencies -- sklear, torch, tensorflow, numpy, transformers

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

# Instructions from Class
# #PART: setting up hyperparameter
# hyper_params = {'gamma':GAMMA, 'C':C}
# clf.set_params(**hyper_params)
# # model hyperparams
# GAMMA = 0.001
# C = 0.5


train_frac = 0.8
test_frac = 0.1
dev_frac = 0.1

# PART: load dataset -- data from csv, tsv, jsonl, pickle
digits = datasets.load_digits()

# PART: sanity check visualization of the data
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)

# PART: data pre-processing -- to remove some noise, to normalize data, format the data to be consumed by mode
# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))


# PART: define train/dev/test splits of experiment protocol
# train to train model
# dev to set hyperparameters of the model
# test to evaluate the performance of the model
dev_test_frac = 1-train_frac
X_train, X_dev_test, y_train, y_dev_test = train_test_split(
    data, digits.target, test_size=dev_test_frac, shuffle=True
)
X_test, X_dev, y_test, y_dev = train_test_split(
    X_dev_test, y_dev_test, test_size=(dev_frac)/dev_test_frac, shuffle=True
)

# PART: Define the model
# Create a classifier: a support vector classifier


# 1. Set the ranges of Hyder parameter
gamma_list = [0.02, 0.007, 0.003, 0.0004, 0.0002]
c_list = [0.1, 0.4, 0.6, 1, 2, 4, 9, 10, 12] 


best_acc = -1.0
best_model = None
best_h_params = None

for i in gamma_list:
    for j in c_list:
      clf = svm.SVC()

      # PART: setting up hyperparameter
      hyper_params = {'gamma': i, 'C' : j}
      clf.set_params(**hyper_params)

      # PART: Train model
      # Learn the digits on the train subset
      clf.fit(X_train, y_train)

      # PART: Get test set predictions
      # Predict the value of the digit on the test subset
      predicted_in_loop = clf.predict(X_dev)

      cur_acc = metrics.accuracy_score(y_pred=predicted_in_loop, y_true=y_dev)

      if cur_acc > best_acc:
        best_acc = cur_acc
        best_model = clf
        best_h_params = hyper_params
        print("Best Accuracy :"+str(hyper_params))
        print("Best Accuracy Value:" + str(cur_acc))


predicted = best_model.predict(X_test)

# PART: Sanity check of predictions
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")

# PART: Compute evaluation metrics
print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)
