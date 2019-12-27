import os, math, cv2, dlib, warnings, itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing import image

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, learning_curve, train_test_split

from util import *
warnings.filterwarnings('ignore')

# ======================================================================================================================
# Data preprocessing for Task A1 and A2
sample_size = 5000 # Full dataset import
landmark_features_celeba, gender_labels, smiling_labels = extract_features_labels_from_celeba(sample_size)
# ======================================================================================================================
# Task A1
# Initialisation parameters
split_percentage = 80 # 80% training data - 20% testing data split
feature_type = 'landmarks' # Feature extraction type
cv_folds = 5 # Number of cross-validation folds (k)

# Train and test the model with hyper-parameter tuning
X_all, X_train, X_test, y_all, y_train, y_test = get_split_data(landmark_features_celeba, gender_labels, split_percentage, feature_type)
model_A1, acc_A1_train, acc_A1_test, cm_A1 = build_svm_gridcv(X_train, X_test, y_train, y_test, cv_folds)

# Plot learning curve and confusion matrix
task_A1_plot, acc_A1_val = plot_learning_curve(model_A1, 'Task A1 Learning Curve - SVM', X_train, y_train, ylim=None, cv=cv_folds,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5))
plot_confusion_matrix(cm_A1, target_names = ['male', 'female'], title = 'Gender Confusion Matrix', cmap=plt.cm.Blues, normalize=False)

# ======================================================================================================================
# Task A2
# Initialisation parameters
split_percentage = 80 # 80% training data - 20% testing data split
feature_type = 'landmarks' # Feature extraction type
cv_folds = 5 # Number of cross-validation folds (k)

# Train and test the model with hyper-parameter tuning
X_all, X_train, X_test, y_all, y_train, y_test = get_split_data(landmark_features_celeba, smiling_labels, split_percentage, feature_type)
model_A2, acc_A2_train, acc_A2_test, cm_A2 = build_svm_gridcv(X_train, X_test, y_train, y_test, cv_folds)

# Save learning curve and confusion matrix plot figures
task_A2_plot, acc_A2_val = plot_learning_curve(model_A2, 'Task A2 Learning Curve - SVM', X_train, y_train, ylim=None, cv=cv_folds,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5))
plot_confusion_matrix(cm_A2, target_names = ['no smile', 'smile'], title = 'Smile Confusion Matrix', cmap=plt.cm.Blues, normalize=False)

# ======================================================================================================================
# Data preprocessing for Task B1 and B2
feature_type = 'rgb' # Feature extraction type
sample_size = 10000 # Full dataset import
img_features_cartoon_set, eye_color_labels, face_shape_labels = extract_features_labels_from_cartoon_set(feature_type, sample_size)
# ======================================================================================================================
# Task B1
# Initialisation parameters
split_percentage = 80 # 80% training data - 20% testing data split
feature_type = 'rgb' # Feature extraction type
cv_folds = 5 # Number of cross-validation folds (k)

# Train and test the model with hyper-parameter tuning
X_all, X_train, X_test, y_all, y_train, y_test = get_split_data(img_features_cartoon_set, eye_color_labels, split_percentage, feature_type)
model_B1, acc_B1_train, acc_B1_test, cm_B1 = build_svm_gridcv(X_train, X_test, y_train, y_test, cv_folds)

# Save learning curve and confusion matrix plot figures
task_B1_plot, acc_B1_val = plot_learning_curve(model_B1, 'Task B1 Learning Curve - SVM', X_train, y_train, ylim=None, cv=cv_folds,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5))
plot_confusion_matrix(cm_B1, target_names = ['0', '1', '2', '3', '4'], title = 'Eye Colour Confusion Matrix', cmap=plt.cm.Blues, normalize=False)

# ======================================================================================================================
# Task B2
# Initialisation parameters
split_percentage = 80 # 80% training data - 20% testing data split
feature_type = 'landmarks' # Feature extraction type
cv_folds = 5 # Number of cross-validation folds (k)

# Train and test the model with hyper-parameter tuning
X_all, X_train, X_test, y_all, y_train, y_test = get_split_data(img_features_cartoon_set, face_shape_labels, split_percentage, feature_type)
model_B2, acc_B2_train, acc_B2_test, cm_B2 = build_svm_gridcv(X_train, X_test, y_train, y_test, cv_folds)

# Save learning curve and confusion matrix plot figures
task_B2_plot, acc_B2_val = plot_learning_curve(model_B2, 'Task B2 Learning Curve - SVM', X_train, y_train, ylim=None, cv=cv_folds,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5))
plot_confusion_matrix(cm_B2, target_names = ['0', '1', '2', '3', '4'], title = 'Face Shape Confusion Matrix', cmap=plt.cm.Blues, normalize=False)

# ======================================================================================================================
# Print out your results with following format:
print('TA1:{},{},{};TA2:{},{},{};TB1:{},{},{};TB2:{},{},{};'.format(acc_A1_train, acc_A1_val, acc_A1_test,
                                                                    acc_A2_train, acc_A2_val, acc_A2_test,
                                                                    acc_B1_train, acc_B1_val, acc_B1_test,
                                                                    acc_B2_train, acc_B2_val, acc_B2_test))