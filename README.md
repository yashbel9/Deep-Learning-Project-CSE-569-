# Fast And Accurate Deep Network Learning Using Exponential Linear Units(ELUs)
 
We introduce the ”exponential linear unit” (ELU) which speeds up learning in deep neural networks and leads to higher classification accuracies. Like rectified linear units (ReLUs), leaky ReLUs (LRe- LUs) and parametrized ReLUs (PReLUs), ELUs alleviate the vanishing gradient problem via the identity for positive values and used a Convolution Neural Network with MNIST as our standard database for the comparison of the activation functions.

# Tasks 

1. Understanding the mathematics and concepts of the paper.
2. Implementing a neural network using sigmoid as activation function.
3. Implementing a neural network using ReLU and ELU as activation functions.
4. Comparing the classification accuracies of all three models.

Implementation of CNN:
We used a Convolution Neural Network with MNIST as our standard database for the comparison of the activation functions. The architecture of the CNN used was CONV-CONV-POOL-DENSE-DENSE where each dense layer had 128 neurons and both convolution layers had 64 and 32 filters each.Dropout was also added in the network to prevent overfitting of the data. 
There were total of 70,000 images, among which 60,000 were used for training the network and 10,000 were used for testing.
The network combinations used were:

1. With RELU activation, with and without batch normalization
2. With LeakyReLU activation with and without batch normalization
3. With ELU activation.
The convolutional neural network was implemented in Keras, which is a python library and which uses Tensorflow in the backend. Matplotlib was used to plot the graphs and loss for each network.
