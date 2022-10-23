import argparse
import json
import os

import cv2
import mediapipe as mp
import numpy as np

save_img = False

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


def parse_args():
    """
    Get parameters from user
    """
    parser = argparse.ArgumentParser(description='Extract pose')
    parser.add_argument('--vid_path', type=str, help='path to video', required=True)
    parser.add_argument('--out_path', type=str, help='output path to save pose', required=True)

    return parser.parse_args()


def extract_data(args):
    """
    Extract pose from images
    """
    video = cv2.VideoCapture(args.vid_path)

    # prepare dirs
    out_dir = args.out_path
    img_dir = os.path.join(out_dir, "image")
    key_dir = os.path.join(out_dir, "keypoint")
    if not os.path.isdir(img_dir):
        os.makedirs(img_dir)
    if not os.path.isdir(key_dir):
        os.makedirs(key_dir)

    # ground truth
    waves = [[0, 60], [61, 119], [120, 172], [173, 223], [224, 312]]
    waves_LR = [[0, 39], [61, 91], [120, 145], [173, 193], [224, 253]]
    waves_RL = [[40, 60], [92, 119], [146, 172], [194, 223], [254, 312]]
    all_percent = []
    for wave_LR, wave_RL in zip(waves_LR, waves_RL):
        all_percent += list(np.linspace(start=0, stop=1, num=wave_LR[1] - wave_LR[0] + 1))
        all_percent += list(np.linspace(start=1, stop=0, num=wave_RL[1] - wave_RL[0] + 1))

    # pose extraction
    idx = 0
    with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5) as pose:
        while True:
            ret, image = video.read()
            if not ret:
                break
            image_height, image_width, _ = image.shape
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            keypoints = {}
            for data_point, name in zip(results.pose_landmarks.landmark, mp_pose.PoseLandmark):
                keypoints[str(name).split(".")[1]] = [data_point.x, data_point.y, data_point.z, data_point.visibility]
            keypoints['completion'] = all_percent[idx]

            # save images with key points for visualization
            if save_img:
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                cv2.imwrite(f"{img_dir}/{idx:04d}.png", image)

            # write pose data to json
            with open(f"{key_dir}/{idx:04d}.json", "w") as final:
                json.dump(keypoints, final, default=str)
            idx += 1


if __name__ == '__main__':
    args = parse_args()
    extract_data(args)
