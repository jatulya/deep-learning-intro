# Deep Learning for Beginners

Youtube account and Github I followed for this -

Deep Learning with TensorFlow 2.0, Keras by codebasics - https://www.youtube.com/playlist?list=PLeo1K3hjS3uu7CxAacxVndI4bE_o3BDtO 

Github - https://github.com/codebasics/deep-learning-keras-tf-tutorial/tree/master

My notebooks contain mostly the same code + my comments/notes I have noted down for later reference.

## Gradient Descent
This function is used to finding the next best suitable weight to be tried on to get the equation in intermediate epoch runs during backward propagation. It takes the current weight, a fixed learning rate (ususally it is taken as 0.01 so as to avoid weights that are way too diffeent from before) and the derivative of the loss function with respect to the weight. The new bias(intercept) is calculated the same way.
