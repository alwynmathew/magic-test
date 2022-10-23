import argparse
import pickle
import time

import cv2
import mediapipe as mp

wave_count = 0
moved = True
wave_percent = 0


def get_fps(prev_t):
    """
    Calculate frame-per-second
    """
    new_t = time.time()
    fps = int(1 / (new_t - prev_t))
    prev_t = new_t
    return fps, prev_t


def parse_args():
    """
    Get parameters from user
    """
    parser = argparse.ArgumentParser(description='Inference wave models.')
    parser.add_argument('--vid_path', type=str, help='path to video', required=True)

    return parser.parse_args()


def pose2wave(keypoints, model):
    """
    Predict wave from pose points
    """
    global wave_count, moved, wave_percent
    out_np = model.predict([keypoints])

    # left sub-wave action
    if out_np == [3] and moved:
        moved = False
        wave_percent = 50

    # right sub-wave action
    if out_np == [1] and not moved:
        wave_count += 1
        moved = True
        wave_percent = 100

    # mid sub-wave action
    if out_np == [2]:
        wave_percent = 25 if moved else 75

    return wave_count, wave_percent


def inference(args):
    """
    Main inference function
    """
    mp_pose = mp.solutions.pose
    # hand key points
    key_name_in = [
        "LEFT_WRIST", "LEFT_ELBOW", "LEFT_SHOULDER",
        "RIGHT_WRIST", "RIGHT_ELBOW", "RIGHT_SHOULDER",
    ]

    # load classifier
    loaded_model = pickle.load(open('wave_classifier', 'rb'))

    # read video object
    video = cv2.VideoCapture(args.vid_path)
    cv2.namedWindow("Magic", cv2.WINDOW_NORMAL)
    prev_t = 0
    show = True

    # image to wave count
    with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5) as pose:
        while True:
            # get frame
            ret, image = video.read()
            fps, prev_t = get_fps(prev_t)
            if not ret:
                break

            # get pose
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # extract essential pose
            keypoints = []
            for key in key_name_in:
                key_name = getattr(mp_pose.PoseLandmark, key)
                data_point = results.pose_landmarks.landmark[key_name]
                keypoints += [data_point.x, data_point.y]

            # get wave count and percent
            wave_count, wave_percent = pose2wave(keypoints, loaded_model)
            print(f"Wave count: {wave_count}, Wave %: {wave_percent}")

            # live visualization (press "q" key to exit)
            if show:
                cv2.putText(image, f"#wave: {wave_count}", (7, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                cv2.putText(image, f"{fps}fps", (7, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                cv2.imshow('Magic', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print('Video capture stopped!')
                    break


if __name__ == '__main__':
    args = parse_args()
    inference(args)
