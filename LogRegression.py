from numpy import *
import matplotlib.pyplot as plt
import time

def sigmoid(X):
    return 1.0/(1+exp(-X))


def LogRegres(X,Y):  #X shape n*m
    n,m=shape(X)

    x_data=zeros((n,m+1))
    for i in range(n):
        j = m-1
        while j>=0:
            x_data[i,j+1]=X[i,j]
            j=j-1
        x_data[i,0]=1
    # x_data = n*(m+1)  
    # x_data' = (m+1)*n 
    m=m+1
    theta=zeros((m,1))
    #theta = (m+1)*1
    alpha=0.01
    for i in range(1000):
        z=sigmoid(matmul(x_data,theta))-Y  # n*1
        a=transpose(x_data)*z   #  m+1*n  n*1
        theta=theta-alpha*a
    s=-5
    for i in range(1000):
        s=s+0.01
        y=(theta[0]+theta[1]*s)/(-theta[2])
        plt.plot(s,y,'og')
    print(theta[0],theta[1],theta[2])



with open("data.txt","r") as f:
    train_x = []
    train_y = []
    for line in f.readlines():
        lineArr=line.strip().split()
        train_x.append([ float(lineArr[0]), float(lineArr[1])])  
        train_y.append([float(lineArr[2])])  
    train_x = mat(train_x)
    train_y = mat(train_y)
    numSamples=train_x.shape[0]
    plt.plot(1,1)
    for i in range(numSamples):
        if int(train_y[i, 0]) == 0:  
            plt.plot(train_x[i, 0], train_x[i, 1], 'or')  
        elif int(train_y[i, 0]) == 1:  
            plt.plot(train_x[i, 0], train_x[i, 1], 'ob')  
    LogRegres(train_x,train_y)



