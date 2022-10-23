import argparse
import json
import os
from sklearn.neighbors import KNeighborsClassifier
import pickle


def parse_args():
    """
    Get parameters from user
    """
    parser = argparse.ArgumentParser(description='Wave models.')
    parser.add_argument('--data_path', type=str, help='path to pose data', required=True)

    return parser.parse_args()


def classic_knn(args):
    """
    Create pose classifier
    """
    # hand key points
    key_name_in = [
        "LEFT_WRIST", "LEFT_ELBOW", "LEFT_SHOULDER",
        "RIGHT_WRIST", "RIGHT_ELBOW", "RIGHT_SHOULDER",
    ]

    # frames at hand on left/right
    wave_L = [0, 61, 120, 173, 224]
    wave_mid = [19, 76, 135, 183, 238]
    wave_R = [39, 91, 145, 193, 253]

    # created ground truth label
    in_wave = wave_L + wave_mid + wave_R
    gt_label = [1] * len(wave_L)
    gt_label += [2] * len(wave_mid)
    gt_label += [3] * len(wave_R)

    # extract needed key points into list
    inputs = []
    for idx in in_wave:
        path = os.path.join(args.data_path, f"{idx:04d}.json")
        with open(path) as json_data:
            key_points = json.load(json_data)
            key_list_in = []
            for key_dict in key_name_in:
                key_list_in += key_points[key_dict][:2]
            inputs.append(key_list_in)

    # classifier
    classifier = KNeighborsClassifier(n_neighbors=2)
    classifier.fit(inputs, gt_label)

    # test classifier
    path = os.path.join(args.data_path, f"{39:04d}.json")
    with open(path) as json_data:
        key_points = json.load(json_data)
        key_list_in = []
        for key_dict in key_name_in:
            key_list_in += key_points[key_dict][:2]
        print(classifier.predict([key_list_in]))

    # save classifier
    with open('wave_classifier', 'wb') as knnPickle:
        pickle.dump(classifier, knnPickle)


if __name__ == '__main__':
    args = parse_args()
    classic_knn(args)
