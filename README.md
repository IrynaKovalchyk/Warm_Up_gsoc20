# Warm_Up_gsoc20
## Easy :

###	Can you explain what is the convolutional neural network?

#### Answer : 
The name “convolutional neural network” indicates that the network employs a mathematical operation called convolution. Convolutional networks are a specialized type of neural networks that use convolution in place of general matrix multiplication in at least one of their layers. So neural networks have a series of hidden layers, through which they transform the input, which is essentially in the form of a single vector.  Each neuron receives some inputs, performs a dot product and optionally follows it with a non-linearity. 

###	Can you explain what is the transfer learning?
#### Answer : 
When we come to use of neural networks in real life problems usage, we come across a big problem, which is lack of sufficient data. Here comes transfer learning. Transfer learning is a machine learning method where a model developed for a task is reused as the starting point for a model on a second task.

It is a popular approach in deep learning where pre-trained models are used as the starting point on computer vision and natural language processing tasks given the vast compute and time resources required to develop neural network models on these problems and from the huge jumps in skill that they provide on related problems.

###	Can you install TensorFlow in your machine and implement a simple CNN on MNIST by TF estimator API following the official document?
#### Answer : 
Yes, I can install Tensorflow in my machine. However, I'm not able to properly implement the aforementioned following the official document. Official document is not updated for TensorFlow 2.0. feature_columns have been moved to tfdatasets recently, and I tried to go around that, as well as also use 'feature_specs' in tfdatasets. Currently, I'm stuck with the error "got an unexpected keyword argument 'input_layer_partitioner'". Input layer partitioners are supposed to be defined explicitely, and otherwise default to min_max_variable_partitioner, according to the official documentation here (https://tensorflow.rstudio.com/reference/tfestimators/dnn_estimators/). They have however been removed in the latest TensorFlow according to the Python version. I'm not sure that the documentation is updated.

Code is attached here as """"" test_task_part1.R """""


## Medium :

###	What are overfitting?
#### Answer :
Overfitting is a term used in statistics that refers to a modeling error that occurs when a function corresponds too closely to a dataset. As a result, overfitting may fail to fit additional data, and this may affect the accuracy of predicting future observations.

Overfitting happens when a model learns the detail and noise in the training dataset to the extent that it negatively impacts the performance of the model on a new dataset. This means that the noise or random fluctuations in the training dataset is picked up and learned as concepts by the model. The problem is that these concepts do not apply to new datasets and negatively impact the model’s ability to generalize..

###	What are L1 and L2 regularization?
#### Answer :
There are several ways of preventing overfitting. Two of them are L1 and L2 regularization. 

In L2 regularization, we encourage the network to use a little of all of its inputs rather than using some of the inputs a lot. Intuitively, we penalize extreme weight vectors and prefer diffuse ones. This is done by adding a term (1/2)*A*(w^2) to the objective, where A is the regularization strength.

In L1 regularization, we add A*|w| for each weight. It leads the weight vectors to become sparse. We use L1 for sparsity training.

###	What should we do if the loss doesn’t converge?
#### Answer :
If the loss doesn’t converge, we should do one of the following : 
1.	We should try lowering the learning rate. It’s possible, that a higher learning rate may have overshot the global minima.
2.	We should check the magnitudes of gradients to see if they are exploding or vanishing. If they are, we should use an adaptive optimizer.

###	Can you implement a simple CNN without estimator API?
####Answer : 
Yes,Code is attached here as """"" test_task_part2.R """""

