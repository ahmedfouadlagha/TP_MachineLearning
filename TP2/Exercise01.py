#Write a program that can find w0 and w1 for an arbitrary dataset of xn; tn pairs
#using the least squares method
import numpy as np
import math
import random

#Function to generate the dataset
def generate_dataset(n):
    xn = []
    tn = []
    for i in range(n):
        xn.append(random.uniform(1928, 2008))
        tn.append(math.sin(2*math.pi*xn[i]) + random.gauss(0, 0.3))
        print("x{} = {}, t{} = {}".format(i+1,xn[i],i+1,tn[i]))
    return xn,tn

def find_w0_w1(xn, tn):
    # Convert xn and tn to numpy arrays
    xn = np.array(xn)
    tn = np.array(tn)

    # Calculate w0 and w1 using the least squares method
    X = np.vstack((xn, np.ones(len(xn)))).T
    w0, w1 = np.linalg.lstsq(X, tn, rcond=None)[0]

    return w0, w1

#Function to calculate the error
def error(w0,w1,xn,tn):
    error = 0
    for i in range(len(xn)):
        error += (tn[i] - (w0 + w1*xn[i]))**2
    return error

#Function to calculate the gradient
def gradient(w0,w1,xn,tn):
    gradient = np.zeros(2)
    for i in range(len(xn)):
        gradient[0] += -2*(tn[i] - (w0 + w1*xn[i]))
        gradient[1] += -2*xn[i]*(tn[i] - (w0 + w1*xn[i]))
    return gradient

#Function to calculate the new w0 and w1
def new_w(w0,w1,gradient,learning_rate):
    w0 = w0 - learning_rate*gradient[0]
    w1 = w1 - learning_rate*gradient[1]
    return w0,w1

#Function to calculate the new learning rate
def new_learning_rate(learning_rate,gradient):
    learning_rate = learning_rate + 0.0001*gradient[0]**2
    return learning_rate

#Function to calculate the new error
def new_error(w0,w1,xn,tn):
    new_error = 0
    for i in range(len(xn)):
        new_error += (tn[i] - (w0 + w1*xn[i]))**2
    return new_error

#Function to calculate the new gradient
def new_gradient(w0,w1,xn,tn):
    new_gradient = np.zeros(2)
    for i in range(len(xn)):
        new_gradient[0] += -2*(tn[i] - (w0 + w1*xn[i]))
        new_gradient[1] += -2*xn[i]*(tn[i] - (w0 + w1*xn[i]))
    return new_gradient

#Main program
xn,tn = generate_dataset(10)
w0,w1 = find_w0_w1(xn,tn)
print("w0 = {}, w1 = {}".format(w0,w1))
print("Error = {}".format(error(w0,w1,xn,tn)))
gradient = gradient(w0,w1,xn,tn)
print("Gradient = {}".format(gradient))
learning_rate = 0.001
w0,w1 = new_w(w0,w1,gradient,learning_rate)
print("w0 = {}, w1 = {}".format(w0,w1))
learning_rate = new_learning_rate(learning_rate,gradient)
print("Learning rate = {}".format(learning_rate))
print("Error = {}".format(new_error(w0,w1,xn,tn)))
gradient = new_gradient(w0,w1,xn,tn)
print("Gradient = {}".format(gradient))




