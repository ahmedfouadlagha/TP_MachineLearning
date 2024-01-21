import matplotlib.pyplot as plt
import numpy as np
import math
import random

def generate_dataset(n):
    xn = []
    tn = []
    for i in range(n):
        xn.append(random.randint(1928, 2008))
        tn.append(math.sin(2*math.pi*xn[i]) + random.gauss(0, 0.3))
        print("x{} = {}, t{} = {}".format(i+1,xn[i],i+1,tn[i]))
    return xn,tn

def find_w0_w1(xn, tn):
    xn = np.array(xn)
    tn = np.array(tn)
    xn_mean = np.mean(xn)
    tn_mean = np.mean(tn)
    xn_tn_mean = np.mean(xn*tn)
    xn2_mean = np.mean(xn**2)
    #Calculate w0 and w1
    w1 = (xn_tn_mean - xn_mean*tn_mean)/(xn2_mean - xn_mean**2)
    w0 = tn_mean - w1*xn_mean
    return w0, w1

def find_X(xn):
    x = np.array(xn)
    a = 2660
    b = 4.3
    return np.array([np.ones(len(x)),x,np.sin((x-a)/b)]).T

def find_w(xn, tn):
    xn = np.array(xn)
    tn = np.array(tn)
    X = find_X(xn)
    return np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),tn)

def Order_n(xn, tn, n):
    xn = np.array(xn)
    tn = np.array(tn)
    #rescaling
    x0 = xn[0]
    xn = (xn - x0)/4
    X = [np.ones(len(xn))]
    for i in range(1,n+1):
        X.append(xn**i)
    X = np.array(X).transpose()
    w = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),tn)
    #Compute the loss 
    lossOrder = loss(w,X,tn)
    print("the average loss for mens 100m data in Order {} is : {}".format(n,lossOrder/len(xn)))
    prediction 2012
    prediction = 0
    for i in range(n+1):
        prediction = prediction + w[i]*((2012-x0)/4)**i
    
    print("the predicted winning time Order {} for 2012 is : {}".format(i,prediction))
    #plot
    plt.plot(xn,tn,'ro',label = "men dataset")
    xplot = np.linspace(min(xn),max(xn),50)
    yplot = 0
    for i in range(n+1):
        yplot = yplot + w[i]*xplot**i
    titleOrder = ""
    if (n==1):
        titleOrder = "1st order degree"
    else:
        titleOrder = f"{n}th order degree"
    plt.plot(xplot,yplot,label = titleOrder)
    plt.title(titleOrder)
    plt.xlabel("years")
    plt.ylabel("seconds")
    plt.legend(loc="upper right")
    plt.show()
    return lossOrder

#Function to calculate the error (squared loss)
def error(w0,w1,xn,tn):
    return sum((tn[i] - (w0 + w1*xn[i]))**2 for i in range(len(xn)))

def loss (w,X,t):
    return np.dot((t-np.dot(X,w)).T,(t-np.dot(X,w)))

#Main program
#Exercise 01: Write a program that can find w0 and w1 for an arbitrary dataset of xn; tn pairs
xn,tn = generate_dataset(10)
w0,w1 = find_w0_w1(xn,tn)
print("w0 = {}, w1 = {}".format(w0,w1))
print("Error = {}".format(error(w0,w1,xn,tn)))

#Exercise 02
#Olympic women's 100m data
xn_W = [1928, 1932, 1936, 1948, 1952, 1956, 1960, 1964, 1968, 1972, 1976, 1980, 1984, 1988, 1992, 1996, 2000, 2004, 2008]
tn_W = [12.20, 11.90, 11.50, 11.90, 11.50, 11.50, 11.00, 11.40, 11.00, 11.07, 11.08, 11.06, 10.97, 10.54, 10.82, 10.94, 11.12, 10.93, 10.78]

w0_W,w1_W = find_w0_w1(xn_W,tn_W)
if (w1_W < 0):
    print("the linear model that minimizes the squared loss for women is t = {} {} x".format(w0_W,w1_W))
else:
    print("the linear model that minimizes the squared loss is t = {} + {} x".format(w0_W,w1_W))

print("the womans winning time at the 2012 Olympic games is : {} seconds".format(w0_W + w1_W*2012))
print("the womans winning time at the 2016 Olympic games is : {} seconds".format(w0_W + w1_W*2016))

#Exercise 03
#Olympic men's 100m data
xn_M = [1896, 1900, 1904, 1906, 1908, 1912, 1920, 1924, 1928, 1932, 1936, 1948, 1952, 1956, 1960, 1964, 1968, 1972, 1976, 1980, 1984, 1988, 1992, 1996, 2000, 2004, 2008]
tn_M = [12.00, 11.00, 11.00, 11.20, 10.80, 10.80, 10.80, 10.60, 10.80, 10.30, 10.30, 10.30, 10.40, 10.50, 10.20, 10.00, 9.95, 10.14, 10.06, 10.25, 9.99, 9.92, 9.96, 9.84, 9.87, 9.85,9.69]
#the Olympic games when it is predicted for women to run a faster winning time than men.
w0_M,w1_M = find_w0_w1(xn_M,tn_M)
print("the Olympic games when it is predicted for women to run a faster winning time than men is : {}".format(4*(((w0_M - w0_W)/(w1_W - w1_M))//4+1)))
print("the predicted winning times is : {}".format(w0_W + w1_W*((w0_M - w0_W)/(w1_W - w1_M))))

plt.plot(xn_W,tn_W,'bo',label="women dataset")
plt.plot([1928,2600],[w0_W + w1_W*1928,w0_W + w1_W*2600])
plt.plot(xn_M,tn_M,'ro',label="man dataset")
plt.plot([1928,2600],[w0_M + w1_M*1928,w0_M + w1_M*2600])
plt.title("Olympic men's and women's 100m data")
plt.xlabel("years")
plt.ylabel("seconds")
plt.legend(loc="upper right")
plt.show()

#Exercise 04
#Compute the weights vector w for Mens 100m data
X_M = find_X(xn_M)
w_M = find_w(xn_M,tn_M)
#Compute the weights vector w for Womens 100m data
X_W = find_X(xn_W)
w_W = find_w(xn_W,tn_W)
#Compute the loss for Mens 100m data
loss_M = loss(w_M,X_M,tn_M)
print("loss for Mens 100m data is : {}".format(loss_M))
#Compute the loss for Womens 100m data
loss_W = loss(w_W,X_W,tn_W)
print("loss for Womens 100m data is : {}".format(loss_W))
#plot the data set point along-with their corresponding model predictions
plt.plot(xn_W,tn_W,'ro',label="women dataset")
plt.plot(xn_M,tn_M,"bo",label="man dataset")
xplot = np.linspace(1928,2016,50)
plt.plot(xplot,w_W[0]+w_W[1]*xplot+w_W[2]*np.sin((xplot-2660)/4.3))
plt.plot(xplot,w_M[0]+w_M[1]*xplot+w_M[2]*np.sin((xplot-2660)/4.3))
plt.legend(loc="upper right")
plt.show()

#Exercise 05
losses = []
for i in range(8):
    lossOrder=Order_n(xn_M,tn_M,i+1)
    losses.append(lossOrder)
plt.plot([1,2,3,4,5,6,7,8],losses)
plt.title("loss orders")
plt.show()

#Exercise 06
#cross validation for the mens 100m data set oder 1
xn_M_training = xn_M[0:19]
xn_M_validation = xn_M[19:]
tn_M_training = tn_M[0:19]
tn_M_validation = tn_M[19:]
#compute the weights vector w for the training data set
w_M_training = find_w(xn_M_training,tn_M_training)
#compute the loss with validation data set
loss_M_validation = loss(w_M_training,find_X(xn_M_validation),tn_M_validation)
print("the loss with validation data set is : {}".format(loss_M_validation))
#compute the average loss with validation data set
loss_average = loss_M_validation/len(xn_M_validation)
print("the average loss with validation data set is : {}".format(loss_average))
#plot 
plt.plot(xn_M_training,tn_M_training,'.',label = "tarining data")
plt.plot(xn_M_validation,tn_M_validation,'.',label = "validation data")
xplot = np.linspace(1896,2008,50)
plt.plot(xplot,w_M_training[0]+w_M_training[1]*xplot)
plt.title("cross validation")
plt.xlabel("years")
plt.ylabel("seconds")
plt.legend(loc="upper right")
plt.show()

#cross-validation k-fold
#data set
xnMen = [1896, 1900, 1904, 1906, 1908, 1912, 1920, 1924, 1928, 1932, 1936, 1948, 1952, 1956, 1960, 1964, 1968, 1972, 1976, 1980, 1984, 1988, 1992, 1996, 2000, 2004, 2008]
tnMen = [12.00, 11.00, 11.00, 11.20, 10.80, 10.80, 10.80, 10.60, 10.80, 10.30, 10.30, 10.30, 10.40, 10.50, 10.20, 10.00, 9.95, 10.14, 10.06, 10.25, 9.99, 9.92, 9.96, 9.84, 9.87, 9.85,9.69]

#split data set
k=5
xn_Kfold = np.array(xnMen)
tn_Kfold = np.array(tnMen)
xn_Kfold = np.array_split(xn_Kfold, k)
tn_Kfold = np.array_split(tn_Kfold, k)

losses = []
count = 0;
for i in range(k):
    for arr in xn_Kfold:
        if(count != i):
            xn_M_training += arr
            
    xn_M_training = np.concatenate(np.delete(xn_Kfold, i, axis=0))
    xn_M_validation = xn_Kfold[i]
    tn_M_training = np.concatenate(np.delete(tn_Kfold, i, axis=0))
    tn_M_validation = tn_Kfold[i]
    w_M_training = find_w(xn_M_training,tn_M_training)
    loss_M_validation = loss(w_M_training,find_X(xn_M_validation),tn_M_validation)
    losses.append(loss_M_validation)
    print(
        f"the loss with validation data set k = {i + 1} is : {loss_M_validation}"
    )

print(
    f"the average loss with validation data set is : {sum(losses) / len(losses)}"
)
plot
for i in range(k):
    xn_M_training = np.concatenate(np.delete(xn_Kfold, i))
    xn_M_validation = xn_Kfold[i]
    tn_M_training = np.concatenate(np.delete(tn_Kfold, i))
    tn_M_validation = tn_Kfold[i]
    w_M_training = find_w(xn_M_training,tn_M_training)
    plt.plot(xn_M_training,tn_M_training,'.',label = "training data")
    plt.plot(xn_M_validation,tn_M_validation,'.',label = "validation data")
    xplot = np.linspace(min(xnMen),max(xnMen),50)
    plt.plot(xplot,w_M_training[0]+w_M_training[1]*xplot)
    plt.xlabel("years")
    plt.ylabel("seconds")
    plt.legend(loc="upper right")
    plt.title(f"k-fold cross-validation k = {i + 1}")
    plt.show()
    


#k fold validation
#data set
xnMen = [1896, 1900, 1904, 1906, 1908, 1912, 1920, 1924, 1928, 1932, 1936, 1948, 1952, 1956, 1960, 1964, 1968, 1972, 1976, 1980, 1984, 1988, 1992, 1996, 2000, 2004, 2008]
tnMen = [12.00, 11.00, 11.00, 11.20, 10.80, 10.80, 10.80, 10.60, 10.80, 10.30, 10.30, 10.30, 10.40, 10.50, 10.20, 10.00, 9.95, 10.14, 10.06, 10.25, 9.99, 9.92, 9.96, 9.84, 9.87, 9.85,9.69]

#split data set
xn_Kfold = xnMen
tn_Kfold = tnMen
k = 5
xn_Kfold = np.array(xn_Kfold)
tn_Kfold = np.array(tn_Kfold)
xn_Kfold = np.array_split(xn_Kfold, k)
tn_Kfold = np.array_split(tn_Kfold, k)


xn_M_training = np.concatenate(np.delete(xn_M, 2))
print(xn_M_training)

losses = []
for i in range(k):
    xn_M_training = np.concatenate(np.delete(xn_M, i))
    xn_M_validation = xn_M[i]
    tn_M_training = np.concatenate(np.delete(tn_M, i))
    tn_M_validation = tn_M[i]
    w_M_training = find_w(xn_M_training,tn_M_training)
    loss_M_validation = loss(w_M_training,find_X(xn_M_validation),tn_M_validation)
    losses.append(loss_M_validation)
    print(f"the loss with validation data set k = {i+1} is : {loss_M_validation}")
    
print(f"the average loss with validation data set is : {sum(losses)/len(losses)}")
#plot
for i in range(k):
    plt.plot(xn_M_training,tn_M_training,'.',label = "training data")
    plt.plot(xn_M_validation,tn_M_validation,'.',label = "validation data")
    xplot = np.linspace(min(xn),max(xn),50)
    plt.plot(xplot,w_M_training[0]+w_M_training[1]*xplot)
    plt.xlabel("years")
    plt.ylabel("seconds")
    plt.legend(loc="upper right")
    plt.title(f"k-fold cross-validation k = {i+1}")
    plt.show()