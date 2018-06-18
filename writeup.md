# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image3]: ./assets/phase-1.jpg "Recovery Image"
[image4]: ./assets/phase-2.jpg "Recovery Image"
[image5]: ./assets/phase-3.jpg "Recovery Image"
[image6]: ./assets/unfliped.jpg "Normal Image"
[image7]: ./assets/fliped.jpg  "Flipped Image"
[image8]: ./assets/nvida_network.png  "Final Model Architecture"
[image9]: ./assets/post_training_analysis.png  "Post training analysis"
[image10]: ./assets/center.jpg  "Center Lane Driving"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* `model.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode
* `model_nvidia.h5` containing a trained convolution neural network 
* `writeup_report.md` summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model_nvidia.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy



#### 1. An appropriate model architecture has been employed


My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 


My model consists of 5 convolution layers with filters 5x5 and 3x3, 3 fully connected layers.
Please take look at function nvidia @ line 204.


The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 220).



#### 2. Attempts to reduce overfitting in the model

My model uses dropout layers with drop_prob of 0.25 and 0.50 after every layer to combat overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

* The model used an adam optimizer, so the learning rate was not tuned manually.

* Number of epoch was purely determined by emperical analysis of when validation loss starts to rise again.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and fnally center lane driving in the opposite direction.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to first build the pipeline, then work on performance.


First apporach was to make a pipeline.

For that I Trained a linear regression model, with all pixels flattened connected to single output.

 Bare bones network flattened --> single output neuron (Regression)

TRAINING LOSS  : 322129.4235
VALIDATION LOSS: 691929.9062

Car keeps circling clockwise. Not a good model, not even close.Phase-1 accomplished. 

Slowly I introduced following techniques into my model.

1. Is sample data really suppose to make sense?
   check video for sample data, use video.py for that
   ---
   Video is same actually, lot of laps.

2. Bare bones network is correct
   Review
   ---
   It is correct

3. Data preprocessing and augmentation.
   a). Add left and right cam images
       Simply append them in generator
       
   b). Add normalizaition
       Use Lambda layer for this 
  
   c). Add image flips to traning data


   c). Add perpective steering angle.
       add rotation off set to left and right images
       This was meant to keep car in center position when deviated.

       ---
       offset = 0.1
       
   d). Crop car hood
       direct way is to simply slice image but will add overhead of editing drive.py for cropping imgaes.
       Better, added cropping layer in keras model


To combat the overfitting, I modified the model so as to include dropout layer after each layer, with model this deep it is natural to see overfitting.

The final step was to run the simulator to see how well the car was driving around track one. Car would start off really nice but it wasn't much responsive to turns. So it turns out we need more data to tell car how to get of off track and take proper turns.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

My model consists of 5 convolution layers with filters 5x5 and 3x3, 3 fully connected layers.
Please take look at function nvidia @ line 204.

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 220).

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image8]



Also, the drive.py file loads image in RGB but our network was modeled to handle BGR colorspace.
so, I changed drive.py so that it would convert RGB images to BGR before sending to network for steering angle prediction.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image10]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover when it goes off track. These images show what a recovery looks like.

![alt text][image3]
![alt text][image4]
![alt text][image5]


To augment the data sat, I also flipped images and angles thinking that this would generalize model to take turns in either direction, not get bias towards a single direction. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Traning Data: 75804
Validation Data: 18951


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by post training analysis. I used an adam optimizer so that manually training the learning rate wasn't necessary.

![alt text][image9]
