# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/Visualization.png "Visualization"
[image2]: ./examples/distribution "distribution"
[image3]: ./examples/grayscale.png "Grayscaling" 


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ?
Number of training examples = 34799

* The size of the validation set is ?
Number of validation examples = 4410

* The size of test set is ?
Number of testing examples = 12630


* The shape of a traffic sign image is ?
Image data shape = (32, 32, 3)

* The number of unique classes/labels in the data set is ?
Number of classes = 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]
![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because a image with grayscale could save us 3 times space and speed up training, and it won't hurt the accuracy of the model.

Here is an example of a traffic sign image after grayscaling.

![alt text][image3]

As a last step, I normalized the image data because the neural network can converge much faster in this way.



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 5x5	    | 1x1 stride, same padding, outputs 10x10x16     									|
| RELU						|												|
| Max pooling						| 2x2 stride, outputs 5x5x16												|
| Fully connected		| input 400, output 120       									|
| RELU						|												|
| Fully connected						| input 120, output 84												|
| RELU						|												|
| Fully conected						| input 84, output 43												|
| Softmax				|         									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used 30 epochs, batch size as 128, learning rate as 0.001. And I use Adam optimizer.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
totally based on LeNet.

* What were some problems with the initial architecture?
I don't think there is something worth fixing in the architecture. the hyperparameters are with learning rate=0.1, batch size=50, epochs=5. I chose it on intuition, just want to have one to experiment.the model is not converging and the number of epochs seem to be too small.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
the architecture is OK. 

* Which parameters were tuned? How were they adjusted and why?
I increased the epochs to 10,30,50. 
Also I decreased the learning rate to 0.01,0.001,0.0001. I found out it worked best when it was 0.001.
And I introduced dropout technique,and use the dropout rate with 0.9.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
This is a image recognition problem, and convolution layer works fine in this kind of problem.
I implemented the dropout layers in the fully connecte layers, which worked fine because the model was overfitting previously.

If a well known architecture was chosen:
* What architecture was chosen?
LeNet

* Why did you believe it would be relevant to the traffic sign application?
because the LeNet architecture was widely used in image recognition.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
After 50 apochs, the training accuracy is 0.998, the validation accuracy is 0.943, and the test accuracy is 0.921.



