# **Traffic Sign Recognition** 

Shon Katzenberger  
shon@katzenberger-family.com  
November 12, 2017

---

## Assignment

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup/trainhisto.png "Train Data Histogram"
[image2]: ./writeup/validhisto.png "Validation Data Histogram"
[image3]: ./writeup/testhisto.png "Test Data Histogram"
[image4]: ./extra/01.jpg "Traffic Sign 1"
[image5]: ./extra/02.jpg "Traffic Sign 2"
[image6]: ./extra/03.jpg "Traffic Sign 3"
[image7]: ./extra/04.jpg "Traffic Sign 4"
[image8]: ./extra/05.jpg "Traffic Sign 5"
[image9]: ./extra/06.jpg "Traffic Sign 6"
[image10]: ./extra/07.jpg "Traffic Sign 7"
[image11]: ./extra/08.jpg "Traffic Sign 8"
[image12]: ./extra/scaled.png "All scaled"
[image13]: ./extra/scored.png "All scored"

## Submission Details

The core submission is the notebook and captured html file, which can be found [here](https://github.com/shonkatzenberger/sdc_proj_2).

---

### Data Set Summary & Exploration

#### Basic data statistics.

I used basic `numpy` operations to get basic summary statistics of the traffic signs data set:

* The size of training set is 34,799.
* The size of the validation set is 4,410.
* The size of test set is 12,630.
* The shape of a traffic sign image is `(32, 32, 3)`.
* The number of unique classes/labels in the data set is 43.

#### Data exploration.

In this section, I loaded the sign names, selected a random representative for each class from the training data,
and displayed them, together with the sign name and number of training examples for that class. These representative
images were helpful when finding signs on the web that should match the various classes.

I also displayed the number of training, validation, and testing examples in each class in histogram form.

**Train Data Histogram:**

![alt text][image1]

**Validation Data Histogram:**

![alt text][image2]

**Test Data Histogram:**

![alt text][image3]

It's apparent that the distributions are not uniform, but the distributions of the three data sets are similar.

### Data Preparation, Model Architecture, and Training

#### 1. Data Preprocessing

**Normalization**

I experimented with different color spaces, namely `RGB`, `YUV`, `Lab`, and grayscale. I used `cv2.cvtColor(...)` to convert the images.
In each case, I then subtracted `128` and divided by `128` to map all features to the half open interval `[-1, +1)`.

Note that the first channel of `YUV` and `Lab` are essentially the same as grayscale, so experimenting with grayscale was just to determine
whether the extra color information of `YUV` and `Lab` was detrimental, perhaps by encouraging over-fitting. In my experiments, grayscale
did not perform as well as the other three. This isn't surprising to me, since I would expect color information to be useful in sign classification.
I was surprised to read in [LeCun's paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) that grayscale performed better for him
than full color did.

I didn't observe a significant difference between the accuracies achieved with the three full-color spaces, so I settled on `RGB` since that
requires the least pre-processing, which could be a small time saving when the model is deployed (although the computation cost of the DNN
would certainly dwarf any color space conversion).

Generally, mapping the features to `[-1, +1)` is theoretically unnecessary in the case when there is no L1/L2 regularization term in the loss function,
or when the features are all in the same scale. This is because the DNN's initial layer weights and biases could simply be adjusted accordingly.
However, since we initialize the weights using the same mean and standard deviation for all layers, without the normalization, it can take the DNN extra epochs
to overcome the poorly scaled initial weights. This would be more significant if we used `sigmoid` activation instead of `relu`, because of the
near-zero derivative of `sigmoid` away from zero. Note that centering the values (subtracting 128) has slight additional value when using padding
(as I do here) in that it makes padding pixels effectively medium gray, rather than black, which seems a bit less detrimental. Note that using
small convolution kernel sizes (as I do here) tends to limit detrimental affects of padding.

**Data Augmentation**

I experimented with boosting the number of examples for under-represented classes by simply replicating examples. There is code for this
in the notebook, together with full comments. This didn't seem to improve validation, so the code isn't being invoked in the submitted
notebook. Of course, using this code increased the time for each epoch, so the net effect was negative.

I did not experiment with other data augmentation techniques, such as flipping images (which would not be appropriate on all images and could
affect the labels), image enhancement / degradation, skewing, random cropping, etc. It wouldn't be difficult to do so, I just didn't have the
time to pursue, and have dealt with it in other settings.

#### 2. DNN Topology

The DNN/CNN topology is implemented in the `BuildDnn` function.

I used a straight-forward DNN/CNN topology consisting of four convolutional layers, each with `3 x 3` kernels, stride one, using padding,
followed by two fully connected layers (counting the final logit layer). I used `relu` for activation. The spatial dimensions were reduced
using max pooling with `2 x 2` patches and stride two, applied after three of the four convolutional layers.

The layer sizes, starting from `32 x 32` inputs are printed in the notebook and duplicated here:

```
Layer 'Input' has shape: (?, 32, 32, 3)
Layer 'First Conv + Pool' has shape: (?, 16, 16, 16)
Layer 'Second Conv' has shape: (?, 16, 16, 16)
Layer 'Third Conv + Pool' has shape: (?, 8, 8, 32)
Layer 'Fourth Conv + Pool' has shape: (?, 4, 4, 64)
Layer 'Flattened' has shape: (?, 1024)
Layer 'First Fully Connected' has shape: (?, 512)
Layer 'Logits' has shape: (?, 43)
```

The first dimension (`?`) of each is the batch dimension.

The DNN also has integrated dropout operators in the following locations:
* Before each of the two fully connected layers, using the 'keep rate' specified by the placeholder `g_kr1`.
* Before the final (fourth) convolutional layer, using the 'keep rate' specified by the placeholder `g_kr2`.
* Before the initial convolutional layer, using the `keep rate` specified by the placeholder `g_kr3`.

Using placeholders for the keep rates makes it easy to experiment with different keep rates on different epochs, as well as to disable dropout
(by setting the keep rates to 1) when not training. The actual keep rates used are discussed below, in the training section.

The implementation of dropout for the final convolutional layer is less than ideal. Common practice is to randomly drop a 'channel' for all pixels.
I couldn't find a good way to do that with TensorFlow without baking the batch size into the tensor shapes, which I didn't want to do.
See the comment in the code for more detail.

I started with only three convolutional layers and fewer nodes in the first fully connected layer (128), but couldn't get past about 96.5%
validation accuracy with that topology. Adding the extra convolution (with no pooling) and increasing the size of the first fully connected
layer (to 512), improved the validation accuracy by about 1% to 97.5%.

#### 3. Training

I used the AdamOptimizer and didn't try anything else. Note that I used a placeholder for the learning rate to make it simple to
vary (without having to rebuild any tensorflow graph nodes).

I did a lot of manual hyperparameter sweeping, paying attention to both the training set error and validation set error.

I ended up with multiple sets of epochs, with decreasing learning rates:
* 10 epochs with learning rate 0.00010
* 5 epochs with learning rate 0.00030
* 5 epochs with learning rate 0.00010
* 5 epochs with learning rate 0.00003

The dropout keep rates that I used were 0.50 for the fully connected layers (goverened by `g_kr1`), 0.95 for the last convolutional
layer (governed by `g_kr2`), and 1.00 for for the first convolution layer (governed by `g_kr3`).
The 50% keep rate for the fully connected layers was critical in avoiding overfitting.
Dropout on the input (`g_kr3`) seemed to always hurt, even at 1% dropout rate (`g_kr3` set to `0.99`), so I disabled this dropout by
setting the keep rate to 1.0.

I experimented with batch size. With my initial lower capacity model, decreasing the batch size to 10 improved the validation accuracy, but
each epoch took about 6 times as long (not a great tradeoff). I also experimented with varying the batch size (decreasing it over time).
I finally settled on the higher capacity network with a batch size of 64.

Each training epoch took about 3 seconds on my MSI laptop that has dual NVidia GeForce GTX 1070 GPUs.

Note that the training and validation accuracy vary from run to run, despite the fact that I fully seed
all random number generators. I've seen this before with TensorFlow and haven't yet figured out the
cause of it.

On some runs the latter epochs seem to be pointless, while on others, the final set of
epochs makes real progress. The run being submitted falls more in the former camp, where
the validation accuracy doesn't seem to improve (nor degrade) over the final 5 epochs.

The final model results were:
* Training set accuracy of 1.000
* Validation set accuracy of 0.976
* Test set accuracy of 0.964

#### 4. Discussion of Choices

The AdamOptimizer is a popular one and I didn't see any reason to try anything else.

Recent trends seem to favor deeper networks with smaller convolutional kernels, so I only used `3 x 3` convolutions.
For example, VGG, SqueezeNet, and MobileNet all use `3 x 3` convolutions extensively. The latter two also use
other techniques, such as 1 x 1 convolutions, depth-wise convolution, and forking / concatenating. Of course,
they are also addressing a much more difficult dataset, with larger input images, many more classes, and poorly
cropped input images. I kept it simply and just used `3 x 3` convolution and fully connected layers, with
no forking / concatenating in the DAG.

After writing the code, most of my effort was spent in hyper-parameter adjustment, as well as adjusting the capacity
of the network as described above.

To limit overfitting I used dropout, with tunable keep rates.
Details of tuning parameters are scattered throughout the discussion above, and in the code.

### Test the Model on New Images

#### 1. The Images

For additional model testing I used the following eight images from the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9]
![alt text][image10] ![alt text][image11]

Two of the images contain multiple signs and I used all signs in these, except for the round pedestrian
sign in the sixth image. The reason for omitting this one is that the pedestrian signs in the training
data appear to be triangular, not round (although I didn't inspect all of them). When I did include this
image, the trained classifier gave a low score to 'pedestrian', favoring round sign classes.

I created a data.csv file that specifies labels, as well as how to crop and resize these images
to get proper input for the classifier. Here are the cropped and scaled images (copied from the notebook):

![alt text][image12]

These images are all fairly clear so should be easy to classify. However, several of them contain
'watermarks' that could confuse a classifier.

#### 2. Model Performance on These Images

Here are the images with ground truth label, prediction and top raw score.

![alt text][image13]

As you can see, the predictions are all correct (100% accuracy). For thorough testing, we would need a lot more than ten images.

#### 3. Model Confidence

The notebook contains the top 5 predictions, together with their raw scores and softmax values, for each of the 10 images.

While many people will call the softmax values "probabilities", they really aren't calibrated to be true probabilities. They do sum to
one, but that doesn't make them probabilities in any true sense. Note that multiplying the raw scores by a constant drastically changes
the computed softmax values, so it is trivial to make a classifier appear "more confident" (by multiplying the final weights and bias by
a constant greater than one) or "less confident" (by multiplying the final weights and bias by a positive constant less than one).

Most of the softmax values are either 1.000 or 0.000, indicating that the difference (call it delta)
between the top raw score and the penultimate score is large enough that exp(-delta) is very close to zero.

There is one image where the top two raw scores are fairly close:

```
Top 5 predictions for image 3 with label 23 'Slippery road'
  Class: 23, Score: 29.286413, Probability: 0.961, Name: Slippery road
  Class: 30, Score: 26.079258, Probability: 0.039, Name: Beware of ice/snow
  Class: 19, Score: 9.483726, Probability: 0.000, Name: Dangerous curve to the left
  Class: 29, Score: 5.147949, Probability: 0.000, Name: Bicycles crossing
  Class: 20, Score: 1.895086, Probability: 0.000, Name: Dangerous curve to the right
```
This is curious, because the canonical signs for these classes seem no more similar to each other than
they are to other triangular signs like 'Bicycles crossing' (significantly further down the list) or
'Wild animals crossing' (not in the top 5). I wonder if this may be because some of the
training images are mis-labeled, that is, some images with a sliding automobile may be mislabeled as
'Beware of ice/snow', causing the model to generate high scores for both of these classes.

Looking at the top 5 predictions for other images can provide some additional interesting insights.
For example, all of the classes in the list above are triangular signs. This suggests that the model is really
paying attention to shape, which is good.

Here's the full results, followed by some additional observations:

```
Top 5 predictions for image 0 with label 17 'No entry'
  Class: 17, Score: 49.800625, Probability: 1.000, Name: No entry
  Class: 14, Score: 15.706025, Probability: 0.000, Name: Stop
  Class: 9, Score: 8.059810, Probability: 0.000, Name: No passing
  Class: 0, Score: 2.655849, Probability: 0.000, Name: Speed limit (20km/h)
  Class: 22, Score: 0.023133, Probability: 0.000, Name: Bumpy road
Top 5 predictions for image 1 with label 13 'Yield'
  Class: 13, Score: 49.903358, Probability: 1.000, Name: Yield
  Class: 9, Score: 15.174875, Probability: 0.000, Name: No passing
  Class: 15, Score: 9.246969, Probability: 0.000, Name: No vehicles
  Class: 2, Score: 2.902282, Probability: 0.000, Name: Speed limit (50km/h)
  Class: 3, Score: 1.561353, Probability: 0.000, Name: Speed limit (60km/h)
Top 5 predictions for image 2 with label 5 'Speed limit (80km/h)'
  Class: 5, Score: 19.120239, Probability: 0.999, Name: Speed limit (80km/h)
  Class: 1, Score: 11.713122, Probability: 0.001, Name: Speed limit (30km/h)
  Class: 2, Score: 11.127168, Probability: 0.000, Name: Speed limit (50km/h)
  Class: 0, Score: 7.895446, Probability: 0.000, Name: Speed limit (20km/h)
  Class: 3, Score: 5.960794, Probability: 0.000, Name: Speed limit (60km/h)
Top 5 predictions for image 3 with label 23 'Slippery road'
  Class: 23, Score: 29.286413, Probability: 0.961, Name: Slippery road
  Class: 30, Score: 26.079258, Probability: 0.039, Name: Beware of ice/snow
  Class: 19, Score: 9.483726, Probability: 0.000, Name: Dangerous curve to the left
  Class: 29, Score: 5.147949, Probability: 0.000, Name: Bicycles crossing
  Class: 20, Score: 1.895086, Probability: 0.000, Name: Dangerous curve to the right
Top 5 predictions for image 4 with label 16 'Vehicles over 3.5 metric tons prohibited'
  Class: 16, Score: 43.719360, Probability: 1.000, Name: Vehicles over 3.5 metric tons prohibited
  Class: 9, Score: 23.284674, Probability: 0.000, Name: No passing
  Class: 15, Score: 13.669271, Probability: 0.000, Name: No vehicles
  Class: 41, Score: 0.736199, Probability: 0.000, Name: End of no passing
  Class: 32, Score: -4.657762, Probability: 0.000, Name: End of all speed and passing limits
Top 5 predictions for image 5 with label 25 'Road work'
  Class: 25, Score: 24.912117, Probability: 1.000, Name: Road work
  Class: 27, Score: 7.354095, Probability: 0.000, Name: Pedestrians
  Class: 33, Score: 1.101917, Probability: 0.000, Name: Turn right ahead
  Class: 24, Score: 0.413363, Probability: 0.000, Name: Road narrows on the right
  Class: 26, Score: -0.121683, Probability: 0.000, Name: Traffic signals
Top 5 predictions for image 6 with label 17 'No entry'
  Class: 17, Score: 50.194107, Probability: 1.000, Name: No entry
  Class: 14, Score: 12.471080, Probability: 0.000, Name: Stop
  Class: 9, Score: 4.205602, Probability: 0.000, Name: No passing
  Class: 23, Score: 1.086516, Probability: 0.000, Name: Slippery road
  Class: 0, Score: 0.864839, Probability: 0.000, Name: Speed limit (20km/h)
Top 5 predictions for image 7 with label 39 'Keep left'
  Class: 39, Score: 37.725521, Probability: 1.000, Name: Keep left
  Class: 40, Score: 20.082232, Probability: 0.000, Name: Roundabout mandatory
  Class: 33, Score: 18.359850, Probability: 0.000, Name: Turn right ahead
  Class: 37, Score: 17.656765, Probability: 0.000, Name: Go straight or left
  Class: 34, Score: 11.702733, Probability: 0.000, Name: Turn left ahead
Top 5 predictions for image 8 with label 12 'Priority road'
  Class: 12, Score: 42.434284, Probability: 1.000, Name: Priority road
  Class: 14, Score: 8.945109, Probability: 0.000, Name: Stop
  Class: 17, Score: 5.993159, Probability: 0.000, Name: No entry
  Class: 42, Score: 5.074652, Probability: 0.000, Name: End of no passing by vehicles over 3.5 metric tons
  Class: 10, Score: 1.819171, Probability: 0.000, Name: No passing for vehicles over 3.5 metric tons
Top 5 predictions for image 9 with label 4 'Speed limit (70km/h)'
  Class: 4, Score: 46.335018, Probability: 1.000, Name: Speed limit (70km/h)
  Class: 0, Score: 24.749758, Probability: 0.000, Name: Speed limit (20km/h)
  Class: 1, Score: 9.680217, Probability: 0.000, Name: Speed limit (30km/h)
  Class: 18, Score: -1.497452, Probability: 0.000, Name: General caution
  Class: 26, Score: -3.363060, Probability: 0.000, Name: Traffic signals
```

For the first sign, 'Stop' is a very reasonable second choice, given the roundish shape and quantity of red. The other
two significant scorers are also round with at least some red.

For the third sign ('Speed limit (80km/h)'), the other choices are all two-digit speed limits. Also note that the
runner up is '30', which, in a poor quality image, could easily be confused with '80'.

For the fifth sign ('Vehicles over ...'), the top two runner ups are round with red borders.

For the ninth image ('Priority road'), the runner up has a significantly lower score (42 vs 9), which reflects
how distinctive this sign is.

For the tenth image ('Speed limit (70km/h)'), the two reasonable alternatives are both speed limit signs, with
'20' being the most likely number to be confused with '70'.

In summary, the top 5 scores on these examples demonstrate reasonable behavior.

### Visualizing the Neural Network

The visual display of the various layers suggests a couple things:
* The first convolution is clearly doing some discrete derivatives, as expected. These pick up
edges in various directions.
* Looking at the second set of images (the pool after the first convolution), I'm struck with how much
information appears to be lost. Perhaps it would be better to have two convolutional layers before pooling,
since we're starting with such low resolution images. That's an experiment for another day.
* Some of the 'feature maps' appear to be worthless (the all-black ones in the images). To know for sure,
we'd need to view these feature maps across multiple classes.
