import os, math, cv2, dlib, warnings, itertools, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing import image

from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, learning_curve, train_test_split

from util import *
warnings.filterwarnings('ignore')

# ======================================================================================================================
# Data import for Task A1 and A2
sample_size = 5000 # Full dataset import
landmark_features_celeba, gender_labels, smiling_labels = extract_features_labels_from_celeba(sample_size, testset=False)
landmark_features_celeba_test, gender_labels_test, smiling_labels_test = extract_features_labels_from_celeba(sample_size, testset=True)
# ======================================================================================================================
# Task A1 - Gender Recognition
# Initialisation parameters
split_percentage = 80 # 80% training data - 20% testing data split
feature_type = 'landmarks' # Feature extraction type
cv_folds = 5 # Number of cross-validation folds (k)

# Train and test the model with hyper-parameter tuning
X_all, X_train, X_test, y_all, y_train, y_test = data_preprocessing(landmark_features_celeba, gender_labels, split_percentage, feature_type)
model_A1, acc_A1_train, acc_A1_test, cm_A1 = build_model_task_A1(X_train, X_test, y_train, y_test)
# model_A1, acc_A1_train, acc_A1_test, cm_A1 = build_svm_gridcv(X_train, X_test, y_train, y_test, cv_folds)

# Plot learning curve and confusion matrix
task_A1_plot, acc_A1_val = plot_learning_curve(model_A1, 'Task A1 Learning Curve - SVM', X_train, y_train, ylim=None, cv=cv_folds,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5))
plot_confusion_matrix(cm_A1, target_names = ['male', 'female'], title = 'Gender Confusion Matrix', cmap=plt.cm.Blues, normalize=False)

# Print results
print('TA1:{},{},{};'.format(acc_A1_train, acc_A1_val, acc_A1_test))

# ======================================================================================================================
# Task A2 - Smiling Recognition
# Initialisation parameters
split_percentage = 80 # 80% training data - 20% testing data split
feature_type = 'landmarks' # Feature extraction type
cv_folds = 5 # Number of cross-validation folds (k)

# Train and test the model with hyper-parameter tuning
X_all, X_train, X_test, y_all, y_train, y_test = data_preprocessing(landmark_features_celeba, smiling_labels, split_percentage, feature_type)
model_A2, acc_A2_train, acc_A2_test, cm_A2 = build_model_task_A2(X_train, X_test, y_train, y_test)
# model_A2, acc_A2_train, acc_A2_test, cm_A2 = build_svm_gridcv(X_train, X_test, y_train, y_test, cv_folds)

# Save learning curve and confusion matrix plot figures
task_A2_plot, acc_A2_val = plot_learning_curve(model_A2, 'Task A2 Learning Curve - SVM', X_train, y_train, ylim=None, cv=cv_folds,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5))
plot_confusion_matrix(cm_A2, target_names = ['no smile', 'smile'], title = 'Smile Confusion Matrix', cmap=plt.cm.Blues, normalize=False)

# Print results
print('TA2:{},{},{};'.format(acc_A2_train, acc_A2_val, acc_A2_test))

# ======================================================================================================================
# Data import for Task B2
feature_type = 'landmarks' # Feature extraction type
sample_size = 10000 # Full dataset import  
img_features_cartoon_set, _, face_shape_labels = extract_features_labels_from_cartoon_set(feature_type, sample_size, testset=False)
img_features_cartoon_set_test, _, face_shape_labels_test = extract_features_labels_from_cartoon_set(feature_type, sample_size, testset=True)
# ======================================================================================================================
# Task B2 - Face Shape Recognition
# Initialisation parameters
split_percentage = 80 # 80% training data - 20% testing data split
feature_type = 'landmarks' # Feature extraction type
cv_folds = 5 # Number of cross-validation folds (k)

# Train and test the model with hyper-parameter tuning
X_all, X_train, X_test, y_all, y_train, y_test = data_preprocessing(img_features_cartoon_set, face_shape_labels, split_percentage, feature_type)
model_B2, acc_B2_train, acc_B2_test, cm_B2 = build_model_task_B2(X_train, X_test, y_train, y_test)
# model_B2, acc_B2_train, acc_B2_test, cm_B2 = build_svm_gridcv(X_train, X_test, y_train, y_test, cv_folds)

# Save learning curve and confusion matrix plot figures
task_B2_plot, acc_B2_val = plot_learning_curve(model_B2, 'Task B2 Learning Curve - SVM', X_train, y_train, ylim=None, cv=cv_folds,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5))
plot_confusion_matrix(cm_B2, target_names = ['0', '1', '2', '3', '4'], title = 'Face Shape Confusion Matrix', cmap=plt.cm.Blues, normalize=False)

# Print results
print('TB2:{},{},{};'.format(acc_B2_train, acc_B2_val, acc_B2_test))

# ======================================================================================================================
# Data import for Task B1
feature_type = 'rgb' # Feature extraction type
sample_size = 4000 
img_features_cartoon_set, eye_color_labels, _ = extract_features_labels_from_cartoon_set(feature_type, sample_size, testset=False)
img_features_cartoon_set_test, eye_color_labels_test, _ = extract_features_labels_from_cartoon_set(feature_type, sample_size, testset=True)
# ======================================================================================================================
# Task B1 - Eye Colour Recognition
# Initialisation parameters
split_percentage = 80 # 80% training data - 20% testing data split
feature_type = 'rgb' # Feature extraction type
cv_folds = 5 # Number of cross-validation folds (k)

# Train and test the model with hyper-parameter tuning
X_all, X_train, X_test, y_all, y_train, y_test = data_preprocessing(img_features_cartoon_set, eye_color_labels, split_percentage, feature_type)
model_B1, acc_B1_train, acc_B1_test, cm_B1 = build_model_task_B1(X_train, X_test, y_train, y_test)
# model_B1, acc_B1_train, acc_B1_test, cm_B1 = build_svm_gridcv(X_train, X_test, y_train, y_test, cv_folds)

# Save learning curve and confusion matrix plot figures
task_B1_plot, acc_B1_val = plot_learning_curve(model_B1, 'Task B1 Learning Curve - SVM', X_train, y_train, ylim=None, cv=cv_folds,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5))
plot_confusion_matrix(cm_B1, target_names = ['0', '1', '2', '3', '4'], title = 'Eye Colour Confusion Matrix', cmap=plt.cm.Blues, normalize=False)

# Print results
print('TB1:{},{},{};'.format(acc_B1_train, acc_B1_val, acc_B1_test))

# ======================================================================================================================
# Print out your results with following format:
print('TA1:{},{},{};TA2:{},{},{};TB1:{},{},{};TB2:{},{},{};'.format(acc_A1_train, acc_A1_val, acc_A1_test,
                                                                    acc_A2_train, acc_A2_val, acc_A2_test,
                                                                    acc_B1_train, acc_B1_val, acc_B1_test,
                                                                    acc_B2_train, acc_B2_val, acc_B2_test))