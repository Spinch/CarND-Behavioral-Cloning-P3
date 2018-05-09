# Behaviorial Cloning Project

---


## Overview

In this project I've created neural network for car cloning driving behavior. Model itself is located in `model.h5` file. `model.py` file is python code for data loading, data analysis, model construction and training. `drive.py` is a python script for using trained model in simulator. `video.mp4` is an example of car moving over the track by NN control. Finally, `writeup.md` file contains description of data collecting, model architecture and training process.


## installation

To use this code you will require Keras, cv2, sklearn, pandas and some other python libraries. The best way to set environment would be to use anaconda with the next command:

```
conda env create -f environment.yml
```

`environment.yml` file is located in the root of project directory.

## Run

To run the project you should change environment with:

```
source activate carnd-term1
```

go to the project directory and run:

```
python3 ./drive.py ./model.h5
```

to run drive control algorithm. Drive simulator which receive control commands can be found [here](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae46bb_linux-sim/linux-sim.zip) (Linux version).

To retrain model you should use command:

```
python3 ./model.py
```

As the result you will get `h5` file with saved model for each train epoch.

## More info

Algorithm description and other interesting information can be found in `writeup.md` file.
