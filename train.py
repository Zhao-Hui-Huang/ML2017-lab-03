import os
import cv2
import numpy as np
from feature import NPDFeature
from sklearn.model_selection import train_test_split
from ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report


def imagePathList(folder):
    root_list = os.listdir(folder)
    image_path_list = []
    for f in root_list:
        f_path = os.path.join(folder, f)
        if os.path.isdir(f_path):
            sub_list = imagePathList(f_path)
            image_path_list += sub_list
        else:
            file_name, file_ext = os.path.splitext(f)
            if '.png' not in file_ext and '.jpg' not in file_ext and '.bmp' not in file_ext and '.jpeg' not in file_ext and '.PNG' not in file_ext and '.JPG' not in file_ext and '.BMP' not in file_ext and '.JPEG' not in file_ext:
                continue
            image_path_list.append(f_path)
    return image_path_list


def rgb2gray24_24(input_dir, output_dir):
    image_path_list = imagePathList(input_dir)
    for image_path in image_path_list:
        image = cv2.imread(image_path, 0)
        image = cv2.resize(image, (24, 24))
        output_path = image_path.replace(input_dir, output_dir)
        output_path_dir = os.path.dirname(output_path)
        if not os.path.exists(output_path_dir):
            os.makedirs(output_path_dir)
        cv2.imwrite(output_path, image)


if __name__ == "__main__":
    image_input_dir = './datasets/original'
    image_output_dir = './datasets/gray'
    features_save_path = './datasets/feature.npz'
    results_path = './report.txt'

    # convert original image to 24 x 24 gray image.
    rgb2gray24_24(image_input_dir, image_output_dir)

    # extract features
    image_path_list = imagePathList(image_output_dir)
    num_of_image = len(image_path_list)
    n_pixels = 24 * 24
    feature_dim = n_pixels * (n_pixels - 1) // 2
    X = np.zeros((num_of_image, feature_dim), dtype=np.float32)
    y = np.zeros((num_of_image,), dtype=np.int32)
    for i, image_path in enumerate(image_path_list):
        image = cv2.imread(image_path, 0)
        image = np.array(image)
        feature = NPDFeature(image).extract()
        if 'nonface' in image_path:
            y[i] = -1
        else:
            y[i] = 1
        X[i, :] = feature
    np.savez(features_save_path, X, y)

    # load features
    npzfile = np.load(features_save_path)
    X = npzfile['arr_0']
    y = npzfile['arr_1']

    # Split the dataset into training set and validation set
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.44, random_state=42, shuffle=True)
    adaboost_classifier = AdaBoostClassifier(max_number_classifier = 1)
    # Train the model
    adaboost_classifier.fit(X_train, y_train)
    y_predict = adaboost_classifier.predict(X_validation)
    target_names = ['non_face', 'face']
    accuracy = np.mean(y_predict == y_validation)
    print(accuracy)
    results = classification_report(y_validation, y_predict, target_names = target_names)
    with open(results_path, 'w+') as f:
        f.write(results)
    print(results)
