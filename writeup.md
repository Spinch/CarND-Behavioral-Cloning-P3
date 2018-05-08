# **Behavioral Cloning** 

In this project I've created neural network to drive car on track in simulator. Network was trained on images captured with human driver car operation.

---

## Data collecting, augmentation and preprocessing

First of all, I've started with getting drive data. NN won;t drive better than human driver it used to learn, so I did my best to get nice data.

![data collecting][image_data_collecting]

I've collected three types of data:

* One lap in forward direction with car on center of the track

* One lap in opposite direction with car on center of the track

* Some examples of car returning to track center from boards

Later I've added 4th set of data, as my trained model behavior was terrible on the bridge:

* Car moving over the bridge

The useful part of data consists of:

* images from center camera:

    ![center][image_example_center]

* images from left camera:

    ![left][image_example_left]

* images from right camera:

    ![right][image_example_right]

* wheel turn angle

Here is the angles distribution for this dataset:

![distribution before augmentation][image_distribution_before]

To feed my network with better data quality, I've done:

1. Added flipped images and reversed angle sign to the dataset
1. Added images from left camera with angle value increased by random value in range [0.,0.2]
1. Added images from right camera with angle value decreased by random value in range [0.,0.2]
1. Data correspond to straight movement was used with 0.5 weight during training

After this I've got much better angles data distribution (with respect to is't weight):

![distribution after augmentation][image_distribution_after]

As the final step of data preproccessing, image cropping and normalization was added as first to layers of Keras model. Image was cropped 60 pixels on top and 20 pixels on bottom. All values was normalized to `[-0.5,0.5]` range.

## Model design and architecture

I've chosen nvidia model from [this article](https://review.udacity.com/#!/rubrics/432/view) as the start point. If it worked on the real car, it have to work on the simulator. But as the simulated world is simpler then the real one and I don't have so much data, I've added some dropout to exclude overfitting possibility.

The result model architecture:

```
| Layer (type)                     | Output Shape          | Param #  | Connected to             |
|:---------------------------------|:----------------------|:---------|:-------------------------|
| cropping2d_1 (Cropping2D)        | (None, 80, 320, 3)    | 0        | cropping2d_input_1[0][0] |
| lambda_1 (Lambda)                | (None, 80, 320, 3)    | 0        | cropping2d_1[0][0]       |
| convolution2d_1 (Convolution2D)  | (None, 38, 158, 24)   | 1824     | lambda_1[0][0]           |
| convolution2d_2 (Convolution2D)  | (None, 17, 77, 36)    | 21636    | convolution2d_1[0][0]    |
| dropout_1 (Dropout)              | (None, 17, 77, 36)    | 0        | convolution2d_2[0][0]    |
| convolution2d_3 (Convolution2D)  | (None, 7, 37, 48)     | 43248    | dropout_1[0][0]          |
| dropout_2 (Dropout)              | (None, 7, 37, 48)     | 0        | convolution2d_3[0][0]    |
| convolution2d_4 (Convolution2D)  | (None, 5, 35, 64)     | 27712    | dropout_2[0][0]          |
| dropout_3 (Dropout)              | (None, 5, 35, 64)     | 0        | convolution2d_4[0][0]    |
| convolution2d_5 (Convolution2D)  | (None, 3, 33, 64)     | 36928    | dropout_3[0][0]          |
| dropout_4 (Dropout)              | (None, 3, 33, 64)     | 0        | convolution2d_5[0][0]    |
| flatten_1 (Flatten)              | (None, 6336)          | 0        | dropout_4[0][0]          |
| dense_1 (Dense)                  | (None, 128)           | 811136   | flatten_1[0][0]          |
| dropout_5 (Dropout)              | (None, 128)           | 0        | dense_5[0][0]            |
| dense_2 (Dense)                  | (None, 64)            | 8256     | dropout_1[0][0]          |
| dropout_6 (Dropout)              | (None, 64)            | 0        | dense_2[0][0]            |
| dense_3 (Dense)                  | (None, 1)             | 65       | dropout_6[0][0]          |

Total params: 950,805
Trainable params: 950,805
Non-trainable params: 0
```

Relu activation function was used for all layers. I've experimented with elu function but was unlucky with it.

## Model training

Adam optimizer was used to decrease hyperparameters number.

I've tried different batch sizes and settled on 128 value.

I've trained my model for 10 epochs but used `ModelCheckpoint` callback from Keras library to save model after each epoch to choose the best one.

I've tried to train model with `fit_generator` method with appropriate generator but as I've performed training on CPU it appeared to be more efficient to load all data into the memory and use regular `fit` method.

## Results

The result you can see in video file `video.mp4`. Car passes one lap of the track without leaving track borders.

---

## Rubric Points

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.


### Required Files

#### 1. The submission includes a model.py file, drive.py, model.h5 a writeup report and video.mp4.

In project repository you can find files:

* model.py - data loading, Keras model and it's training
* drive.py - script to drive a car in simulator
* model.h5 - saved trained network
* writeup.md - this file
* video.mp4 - example of car operated by trained NN

### Quality of Code

#### 1. The model provided can be used to successfully operate the simulation.

Model saved in `model.h5` can be used with driver script by running:

```
python3 ./drive.py ./model.h5
```

#### 2. The code in model.py uses a Python generator, if needed, to generate data for training rather than storing the training data in memory. The model.py code is clearly organized and comments are included where needed.

I've tried to use Pyhton generator and Kears `fit_generator` function and it worked well, but it appeared for me to be easier to train model on CPU and to load all data in the memory at training startup.

The code is split for different functions:

* data load
* data analyzes
* data prepocess
* generator
* model architecture creation
* main block

Code is commented.

### Model Architecture and Training Strategy

#### 1. The neural network uses convolution layers with appropriate filter sizes. Layers exist to introduce nonlinearity into the model. The data is normalized in the model.

I use [nvidia network architecture ](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) with 5 convolution layers with 3x3 and 5x5 filter sizes and depth from 24 to 64.

I use relu activation function (see code line 109) to introduce nonlinerity into the model. I've experimented with elu activation function but it haven't shown good performance.

Data is normalized with Keras lamda layer (see code line 121).

#### 2. Train/validation/test splits have been used, and the model uses dropout layers or other methods to reduce overfitting.

Data split to train and validation sets is performed by `sklearn.model_selection.train_test_split` function (see line 179).

Dropout layers are used to reduce model ovefitting (see lines 134, 138, 142, 146, 150,152).

#### 3. Learning rate parameters are chosen with explanation, or an Adam optimizer is used.

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Training data has been chosen to induce the desired behavior in the simulation (i.e. keeping the car on the track).

That's how I've chosen training data:

1. One lap in forward direction with car on center of the track
1. One lap in opposite direction with car on center of the track
1. Some examples of car returning to track center from boards
1. Some more data fro challenging environment (bridge)

### Architecture and Training Documentation

#### 1. The README thoroughly discusses the approach taken for deriving and designing a model architecture fit for solving the given problem.

It was described earlier in "Model design and architecture" section.

#### 2. The README provides sufficient details of the characteristics and qualities of the architecture, such as the type of model used, the number of layers, the size of each layer. Visualizations emphasizing particular qualities of the architecture are encouraged.

It was described earlier in "Model design and architecture" section.

#### 3. The README describes how the model was trained and what the characteristics of the dataset are. Information such as how the dataset was generated and examples of images from the dataset must be included.

It was described earlier in "Data collecting, augmentation and preprocessing" and "Model training" sections.

### Simulation

#### 1. No tire may leave the drivable portion of the track surface. The car may not pop up onto ledges or roll over any surfaces that would otherwise be considered unsafe (if humans were in the vehicle).

The car drive manner fits this criteria, what can bee seen in video file `video.mp4`.


[//]: # (Image References)

[image_data_collecting]: ./writeup_img/dataColProc.jpg "Data collection process"
[image_distribution_before]: ./writeup_img/dist1.png "Data distribution before preprocessing"
[image_distribution_after]: ./writeup_img/dist2.png "Data distribution after preprocessing"
[image_example_center]: ./writeup_img/example_c.jpg "Example from center camera"
[image_example_left]: ./writeup_img/example_l.jpg "Example from left camera"
[image_example_right]: ./writeup_img/example_r.jpg "Example from right camera"

