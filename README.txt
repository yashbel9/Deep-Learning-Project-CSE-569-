Requirements: 
Python 3 
Keras
Matplotlib
pickle


The file contains 6 python files which have 5 different models, and one code to plot the graphs.

1. keras_leakyrelu_batchnorm.py -    CNN with LeakyRelu activation, with batch normalization 
2. keras_relu_batchnorm.py -         CNN with RELU activation, with batch normalization 
3. keras_mnist_cnn_elu.py -          CNN with ELU activation 
4. keras_mnist_cnn.py -              CNN with RELU activation 
5. keras_mnist_cnn_leakyrelu.py -    CNN with Leaky Relu activation
6. plot.py                           Plotting the graphs.


To run the file:
python <filename>


Instructions. 
1. First run all the models one by one. The model saves the model parameters like accuracy, loss in the same file as a .pickle file 
2. Run the plot.py that uses the .pickle files, and saves the graphs as .png in the same file. 



