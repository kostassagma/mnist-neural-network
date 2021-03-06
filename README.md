# MNIST neural network

This is a effort to solve the "hello world" of neural networks. This is also my first project that involves neural networks and machine learning in general.

## How to run
```
$ git clone https://github.com/kostassagma/mnist-neural-network.git
$ cd mnist-neural-networks
$ python -m venv venv
$ venv/Scripts/activate
$ pip install -r requirements.txt
```

## Network Structure
The neural network consists of 4 layers. At first, there is an input layer, which converts 28 by 28 images into 784 consecutive inputs. Then, there are two hidden layers with 256 neurons each that use the relu activation function and, at last, the output layer consists of ten neurons, each representing a digit, and thus expressing how much our neural network thinks the respective digit is displayed in each image. The structure can be seen below:

![Structure](/for-readme/structure.png)

## Activation Fucntions
Both hidden layers use the relu activation function (rectified linear unit). Its charasteristic property is that is returns 0 for values smaller than zero and is linear for values greater than zero, thus resembling the behaviour of biological neurons and either being inactive or having a value proportionate to its stimulation.

![Relu](/for-readme/relu.png)

The output layer makes use of a softmax function. This way, each layer can produce binary values. These binary values express how likely each image is to correspond to each digit

![Softmax](/for-readme/softmax.png)
