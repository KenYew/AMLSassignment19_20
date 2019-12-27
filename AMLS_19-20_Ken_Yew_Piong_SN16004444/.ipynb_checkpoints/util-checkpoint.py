import os, math, cv2, dlib, itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing import image

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, learning_curve

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

#======================================================================
# Data Pre-processing Functions
#======================================================================
def get_split_data(X, y, split_percentage, feature_type):
    
    Y = np.array([y, -(y - 1)]).T
    split = int(len(X) * (split_percentage/100))
    X_all = X
    y_all = Y
    X_train = X[:split]
    y_train = Y[:split]
    X_test = X[split:]
    y_test = Y[split:]
    
    if feature_type == 'rgb': 
        X_all = np.reshape(X_all, (X_all.shape[0], -1))
        X_train = np.reshape(X_train, (X_train.shape[0], -1))
        X_test = np.reshape(X_test, (X_test.shape[0], -1))
    elif feature_type == 'landmarks':
        X_all = X_all.reshape(len(X_all), 68*2)
        X_train = X_train.reshape(len(X_train), 68*2)
        X_test = X_test.reshape(len(X_test), 68*2)
    
    y_all = list(zip(*y_all))[0]
    y_train = list(zip(*y_train))[0]
    y_test = list(zip(*y_test))[0]
    
    return X_all, X_train, X_test, y_all, y_train, y_test

def scale_data(data):
    scaler = StandardScaler()
    scaler.fit(data)
    scaled_data = scaler.transform(data)
    return scaled_data

#======================================================================
# Feature Extraction Functions
#======================================================================
def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)

def run_dlib_shape(image):
    # in this function we load the image, detect the landmarks of the face, and then return the image and the landmarks
    # load the input image, resize it, and convert it to grayscale
    resized_image = image.astype('uint8')

    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype('uint8')

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    num_faces = len(rects)

    if num_faces == 0:
        return None, resized_image

    face_areas = np.zeros((1, num_faces))
    face_shapes = np.zeros((136, num_faces), dtype=np.int64)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        temp_shape = predictor(gray, rect)
        temp_shape = shape_to_np(temp_shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)],
        #   (x, y, w, h) = face_utils.rect_to_bb(rect)
        (x, y, w, h) = rect_to_bb(rect)
        face_shapes[:, i] = np.reshape(temp_shape, [136])
        face_areas[0, i] = w * h
    # find largest face and keep
    dlibout = np.reshape(np.transpose(face_shapes[:, np.argmax(face_areas)]), [68, 2])

    return dlibout, resized_image

def extract_features_labels_from_celeba(sample_size):
    """
    This function extracts the landmarks features for all images in the folder 'Dataset/celeba'.
    It also extracts the gender and smiling labels for each image.
    :return:
        landmark_features:  an array containing 68 landmark points for each image in which a face was detected
        gender_labels:      an array containing the gender label (male=0 and female=1) for each image in
                            which a face was detected
        smiling_labels:     an array containing the smiling label (not smiling=0 and smiling=1) for each image in
                            which a face was detected
    """
    # Global Parameters
    basedir = './Datasets/celeba'
    images_dir = os.path.join(basedir,'img')
    labels_filename = 'labels.csv'

    # Setting paths of images and labels
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    target_size = None
    labels_file = open(os.path.join(basedir, labels_filename), 'r')
    
    # Obtaining the labels
    lines = labels_file.readlines()
    lines = [line.strip('"\n') for line in lines[:]]
    gender_labels = {line.split('\t')[0] : int(line.split('\t')[2]) for line in lines[1:]}
    smiling_labels = {line.split('\t')[0] : int(line.split('\t')[3]) for line in lines[1:]}
    
    # Extract landmark features and labels
    if os.path.isdir(images_dir):
        all_features = []
        all_gender_labels = []
        all_smiling_labels = []
        for img_path in image_paths[0:sample_size]:
            if not img_path.endswith('.jpg'):
                continue
            file_name= img_path.split('.')[1].split('/')[-1]

            # load image
            img = image.img_to_array(
                image.load_img(img_path,
                               target_size=target_size,
                               interpolation='bicubic'))
            features, _ = run_dlib_shape(img)
            if features is not None:
                all_features.append(features)
                all_gender_labels.append(gender_labels[file_name])
                all_smiling_labels.append(smiling_labels[file_name])

    landmark_features = np.array(all_features)
    gender_labels = (np.array(all_gender_labels) + 1)/2 # simply converts the -1 into 0, so male=0 and female=1
    smiling_labels = (np.array(all_smiling_labels) + 1)/2
    return landmark_features, gender_labels, smiling_labels

def extract_features_labels_from_cartoon_set(feature_type, sample_size): 
    """
    This function extracts the features for all images in the folder 'Dataset/cartoon_set'.
    It also extracts the eye color and face shape labels for each image.
    :input:
        feature_type: select whether to extract landmark or pixel features
    :return:
        img_features:  an array containing 68 landmark points or 256x256 pixel data for each image in which a face was detected
        eye_color_labels:   an array containing the eye color labels for each image in
                            which a face was detected
        face_shape_labels:  an array containing the face shape labels for each image in
                            which a face was detected
    """
    # Global Parameters
    basedir = './Datasets/cartoon_set'
    images_dir = os.path.join(basedir,'img')
    labels_filename = 'labels.csv'

    # Setting paths of images and labels
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    target_size = None
    labels_file = open(os.path.join(basedir, labels_filename), 'r')

    # Obtaining the labels
    lines = labels_file.readlines()
    lines = [line.strip('"\n') for line in lines[:]]
    eye_color_labels = {line.split('\t')[0] : int(line.split('\t')[1]) for line in lines[1:]}
    face_shape_labels = {line.split('\t')[0] : int(line.split('\t')[2]) for line in lines[1:]}

    # Extract landmark features and labels
    if os.path.isdir(images_dir):
        all_features = []
        all_eye_color_labels = []
        all_face_shape_labels = []
        for img_path in image_paths[0:sample_size]:
            if not img_path.endswith('.png'):
                continue
            file_name= img_path.split('.')[1].split('/')[-1]

            if feature_type == 'rgb':
                # Using the imread function to read each image file. The argument 1 means that the image is NOT grayscaled
                img_array = cv2.imread(img_path, cv2.IMREAD_COLOR) 
                # Downsampling the image from 500 x 500 to 128 x 128 so as to reduce training times and speed up hyperparametrization
                features = cv2.resize(img_array, (128, 128))

            elif feature_type == 'landmarks':
                # load image
                img = image.img_to_array(
                    image.load_img(img_path,
                                   target_size=target_size,
                                   interpolation='bicubic'))
                features, _ = run_dlib_shape(img)

            if features is not None:
                all_features.append(features)
                all_eye_color_labels.append(eye_color_labels[file_name])
                all_face_shape_labels.append(face_shape_labels[file_name])

    img_features = np.array(all_features)
    eye_color_labels = np.array(all_eye_color_labels)
    face_shape_labels = np.array(all_face_shape_labels)
    return img_features, eye_color_labels, face_shape_labels

#======================================================================
# Machine Learning Functions
#======================================================================
def build_svm_gridcv(X_train, X_test, y_train, y_test, cv_folds):
    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'],
                         'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10]},
                        {'kernel': ['linear'], 'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10]},
                        {'kernel': ['poly'],
                         'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10]}]

    scores = ['precision'] # 'recall'

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(
            SVC(), tuned_parameters, cv = cv_folds, scoring='%s_macro' % score
        )
        clf.fit(X_train, y_train)
    
        print("Best parameters set found on training dataset:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on training dataset:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full training dataset.")
        print("The scores are computed on the full testing dataset.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()
    
    print('The confusion matrix is:')
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    print()

    print('Best estimator found:', clf.best_estimator_)
    print('Best parameters set found:', clf.best_params_)
    print()
    
    acc_score_test = accuracy_score(y_test, y_pred)
    print('SVM with GridCV on testing data - Accuracy Score: %.3f (+/- %.3f)' % (acc_score_test.mean(), acc_score_test.std()))
    acc_score_train = clf.score(X_train, y_train)
    print('SVM with GridCV on training data - Accuracy Score: %.3f (+/- %.3f)' % (acc_score_train.mean(), acc_score_train.std()))
    print() 
    
    return clf, acc_score_train, acc_score_test, cm

#======================================================================
# Plot Functions
#======================================================================
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
   
    plt.figure(figsize=(8, 6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.savefig(title+'.png')

    return plt, np.mean(train_scores_mean)

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(6, 4))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig(title+'.png')