# Model Results

## HitDetect
### Model architecture
Architecture of our hit detection model is based on a
simple GRU-based recurrent network that consumes court, pose, and
2D shuttlecock information to make hit predictions

![HitDetect](images/network/hit-detect.png)

ref: [MonoTrack: Shuttle trajectory reconstruction from monocular badminton video](https://arxiv.org/pdf/2204.01899v2.pdf)
### Results (V1)

| Appoarches | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|
|Before optimization|0.556|0.315|0.374|**0.677**|
|After optimization|0.937|0.469|0.469|0.471|
|After optimization (tolerance=1)|0.979|0.570|0.654|0.526|
|After Before optimization (tolerance=3)|**0.989**|**0.697**|**0.735**|0.672|

### Results (V2)

**Changes:** Random new negative samples every epoch. 

| Appoarches | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|
|Before optimization|0.575|0.325|0.377|**0.705**|
|After optimization|0.939|0.453|0.469|0.449|
|After optimization (tolerance=1)|0.977|0.533|0.628|0.486|
|After Before optimization (tolerance=3)|**0.987**|**0.678**|**0.722**|0.652|

## ShotDetect
Similar to hit detection model except shot detection adds net kp and the number of consecutive frames = 30

