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

[image1]: ./images/hist.png "Histogram"
[image2]: ./images/sign_sample.png "Sample of dataset"
[image3]: ./images/grayscale_and_normalize.png "Prepare image"
[image4]: ./traffic_signs/sign1.png "Traffic Sign 1"
[image5]: ./traffic_signs/sign2.png "Traffic Sign 2"
[image6]: ./traffic_signs/sign3.png "Traffic Sign 3"
[image7]: ./traffic_signs/sign4.png "Traffic Sign 4"
[image8]: ./traffic_signs/sign5.png "Traffic Sign 5"

---
### Writeup / README


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used standard methods and numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an example of dataset element
![alt text][image2]

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distributed
![alt text][image1]

Evedently there is disbalance in distribution of dataset, we can see that for example that about the first 11-12 traffic signs are represented a lot more that the last traffic signs. In this execise we will not do anything about it. But it's better to balance the representation of lables in dataset.


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because the edges have the crucial importance on the model, while the color is not stable and can change under different curcumstances, so it could be too difficult for this classifier (underfiting) and of couse it would be required the additinal prepearing images, so they would not so much react on ligtning conditions and shadows.   

After graysacaling I decided to normalize images to the range (-1, 1), for more optimized performance of the classifier.
Here is an example of the data sample before and after grayscaling and normalizing 

![alt text][image3]



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:
| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 grayscale image   							|
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 2x2 stride, valid padding, outputs 10x10x16    |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Convolution 1x1	    | 2x2 stride, valid padding, outputs 1x1x412    |
| RELU					|												|
| Fully connected		| input 412, output 122        									|
| RELU					|												|
| Dropout				| 50% keep        									|
| Fully connected		| input 122, output 84        									|
| RELU					|												|
| Dropout				| 50% keep        									|
| Fully connected		| input 84, output 43        									|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used differenc learning rates and different batch sized. Also I tried long training 30 - 50 epochs and more fast traning for 10 - 20 epocs. I picked eventually learning rate 0.001 and batch size 128. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.994
* validation set accuracy of 0.951 
* test set accuracy of 0.928

The first model I tried was simple LeNet model with 8 levels. But it didn't serve to my perposes much
To make the model more generic I tried regularization, regularization could help to simpify and reinfource the model by minimizing of squares of weights, but to get the needed accuracy I had to use a lot of epochs, which was not quite preferable.
So I use two dropout layers to generalize model, which in only 10 epochs could reach quite decent values of accuracy.

 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because the is a lot of prohibitory traffic   signs with the same form and content. 
The second model can be challengeable because it's similar to others signs (round sign with vehicles in the center)
The third sign is speed limit sign, so there is a lot signs with different numbers as content.
Next sign has pretty bad quality and can be difficult to recognize the content of the triangle.
The last sign is pretty straitforward and can be used for comparison of the degree of assurance of the model in it's own output results.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).
Here are the right signs of images:
22, 9, 5, 28, 12

Here are the results of the prediction:
22, 9, 1, 11, 12

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Bumpy Road     		| Bumpy Road   									| 
| No passing   			| No passing 									|
| Speed limit (80km/h)	| Speed limit (30km/h)							|
| Children crossing		| Right-of-way at the next intersection			|
| Priority road			| Priority road      							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%.

We can see that our model had problems with recognition of numbers. The propabilites distribution shows that our model pretty shure that 30 is 80. I think that the reason is lack of enough images in training set with speed limits and figures as content. Also this big percent of assurance can signalized the the model overtrained.  
Children crossing sign fully confused the model. We can't see this sign even in probability distribution. The problem is bad quality and huge amount of signs as triangle with people and other stuff in the center and not representative sign it collection itself. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

image 1:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| Bumpy Road    		|	|	|	|	|	|	|

image 2:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| No passing   									| 
| .01     				| Others 										|

image 3:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Speed limit (30km/h)  						| 
| .01     				| Others 										|

image 4:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .68         			| Right-of-way at the next intersection    		| 
| .06     				| Road work 									|
| .05					| Pedestrians									|
| .04	      			| Dangerous curve to the right					|
| .03				    | Beware of ice/snow  							|

image 5:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| Priority road   								|



