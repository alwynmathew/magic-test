# Magic CV test

Count number of waves from video

Wave action is the combination of the hand moving from left to right or vice versa. As a simple, lightweight 
solution is needed, I used classic K-NN to classify hand poses when hands are at the left and the right location. 
Once the classifier can let us know when the hand reaches left or right, and a wave is recorded whenever a combination 
of left and right hand event happens in sequence. 

## Environment setup

```
pip install -r requirements.txt
```

## Extract pose from images

```
python3 extract_pose.py --vid_path videos/A.mp4 --out_path /path/to/dir
```

`out_path` is the path to directory to store extracted pose from video located at `vid_path`.

## Learn wave pattern

```
python3 wave_classifier.py --data_path /path/to/dir/keypoint
```

`data_path` is the path to keypoint directory extracted in above step.

## Test wave count on video

```
python3 test_wave.py --vid_path videos/B.mp4
```

`vid_path` is the path to the test video.

