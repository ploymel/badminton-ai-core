# BadmintonAI Core
A core repository of badminton AI project.

## Download weights
The model weights can be download at https://drive.google.com/drive/folders/1KjBKXC8qKjJ4tAdL9O71e2-moBwid4rE?usp=drive_link

After downloading the models, make a new directory `weights` and put under `src/models`. The structure should be `src/models/weights`. Then, move all the weights inside the downloaded folder to this directory.

## Prerequisite
You need GPU to run this code. A Gaming notebook should be sufficient (most likely).  
Other requirements:
- python 3.9++
- pytorch related to your gpu. see https://pytorch.org/get-started/locally/ for more information.
- install other python libs using `pip install -r requirements.txt`

## To Run
```bash
python main.py --folder_path "videos" --result_path "res" --force
```

`--folder_path` = video input folder  
`--result_path` = output folder  
`--force` = force run all of the videos in the input folder

## To Visualize the results
```bash
python src/tools/VideoDraw.py --folder_path "videos" --result_path "res" --force --court --net --players --ball --trajectory
```