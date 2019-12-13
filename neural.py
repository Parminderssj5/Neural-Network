import numpy as np
import random
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from scipy.special import expit
import math

train=pd.read_csv("/kaggle/input/mnist-in-csv/mnist_train.csv")
y_train=train['label']
X_train=train.drop(['label'],axis=1)
X_train=np.array(X_train)
y_train=np.array(y_train)
y_train=np.reshape(y_train,(-1,1))
X_train=X_train/255.0
lb = LabelBinarizer()
lb.fit([i for i in range(10)])
y_train=lb.transform(y_train)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_train,y_train,test_size=0.3,random_state=42)

def Layer(num_units,activation,num_units_in_prev_layer):
    l={}
    l['num_units']=num_units
    l['activation']=activation
    l['outputs']=np.zeros(num_units)
    l['thetas']=np.random.randn(num_units,num_units_in_prev_layer)*0.001
    l['inputs']=None
    l['gradients']=None
    return l


def neural_net(num_inputs,num_hidden_units_list,activation):
    nn={}
    if len(num_hidden_units_list)==0:
        nn['hidden_layer_sizes']=num_hidden_units_list
        nn['layers']=[Layer(10,"sigmoid",num_inputs)]
        nn['num_layers']=len(num_hidden_units_list)+1
    else:
        nn['hidden_layer_sizes']=num_hidden_units_list
        nn['num_layers']=len(num_hidden_units_list)+1
        nn['layers']=[Layer(num_hidden_units_list[0],activation,num_inputs)]
        for i in range(1,len(num_hidden_units_list)):
            layer=Layer(num_hidden_units_list[i],activation,num_hidden_units_list[i-1])
            nn['layers'].append(layer)
        layer=Layer(10,"sigmoid",num_hidden_units_list[-1])
        nn['layers'].append(layer)
    return nn


def forward(nn,inp,dropout):
    inp=np.matrix(inp)
    nn['layers'][0]['inputs']=inp
    nn['layers'][0]['netj']=inp @ (np.matrix(nn['layers'][0]['thetas']).T)
    nn['layers'][0]['outputs']=(1-dropout)*apply_acti(nn['layers'][0]['netj'],nn['layers'][0]['activation'])
    for i in range(1,nn['num_layers']):
        layer=nn['layers'][i]
        layer['inputs']=nn['layers'][i - 1]['outputs']
        layer['netj']=layer['inputs'] @ (np.matrix(nn['layers'][i]['thetas']).T)
        if i==nn['num_layers']-1:
            layer['outputs']=apply_acti(layer['netj'],layer['activation'])
        else:
            layer['outputs']=(1-dropout)*apply_acti(layer['netj'],layer['activation'])


def backward(nn,gold):
    out_layer=nn['layers'][-1]
    gold=np.matrix(gold)
    out_layer['grad_wrt_netj']=-np.multiply((gold - out_layer['outputs']),grad(out_layer['netj'],out_layer['activation']))
    out_layer['gradients']=(out_layer['grad_wrt_netj'].T) @ out_layer['inputs']
    for i in range(nn['num_layers'] - 2, -1, -1):
        layer=nn['layers'][i]
        next_layer=nn['layers'][i + 1]
        layer['grad_wrt_netj']=np.multiply((next_layer['grad_wrt_netj'] @ next_layer['thetas']),grad(layer['netj'],layer['activation']))
        layer['gradients']= (layer['grad_wrt_netj'].T) @ layer['inputs']


def apply_acti(output,activation):
    if activation=="sigmoid":
        return sigmoid(output)
    if activation == "relu":
        return relu(output)


def grad(netj, activation):
    if activation == "sigmoid":
        oj=sigmoid(netj)
        return np.multiply(oj,(1 - oj))
    if activation == "relu":
        temp = np.matrix(netj)
        temp[temp < 0] = 0
        temp[temp >= 0] = 1
        return temp


def update_weights(nn,eeta,lmda):
    for layer in nn['layers']:
        layer['thetas']= (1-2*eeta*lmda)*layer['thetas']-eeta*(layer['gradients'])


def error(nn,gold):
    out_layer=nn['layers'][-1]
    e=gold-out_layer['outputs']
    e=np.sum(np.square(e))/len(gold)
    return e


def predict(nn,inp):
    forward(nn,inp,0)
    out=nn['layers'][-1]['outputs']
    return np.array(out.argmax(axis=1)).flatten()


def relu(output):
    output[output<0]=0
    return output


def sigmoid(output):
    return expit(output)


def train(nn,data,labels,eeta=0.01,batch_size=100,max_iter=100,threshold=1e-4,decay=False,dropout=0,lmda=0):
    zip_data=list(zip(data.tolist(),labels.tolist()))
    random.shuffle(zip_data)
    old_error=None
    epochs=1
    lr=eeta
    factor=1
    while(epochs <= max_iter):
        err = 0
        for i in range(0, len(zip_data), batch_size):
            batch=zip_data[i: i + batch_size]
            x,y=zip(*batch)
            forward(nn,np.array(x),dropout)
            err+=error(nn,np.array(y))
            backward(nn,np.array(y))
            if decay:
                update_weights(nn,eeta/math.sqrt(factor),lmda)
            else:
                update_weights(nn,lr,lmda)

        err /= (len(zip_data) / batch_size)
        if epochs == 1:
            print("epoch = 1   error = "+str(round(err,8)))
        else if err > old_error:
     		factor += 1
        old_error = err
        epochs += 1
    print("final error : "+str(round(err,8)))


def getacc(y_pred,y_test):
    n=len(y_test)
    k=0
    for i in range(n):
        for j in range(10):
            if y_pred[i]==j and y_test[i][j]==1:
                k+=1
    return (k/n)


lrates=[0.01,0.05,0.1]
hlayers=[[50],[50,50],[50,50,50],[50,50,50,50]]

for lrate in lrates:
    for hlayer in hlayers:
        my_nn=neural_net(784,hlayer,'sigmoid')
        print("lrate : "+str(lrate)+"  "+"No. of hidden layers : "+str(len(hlayer)))
        train(my_nn,X_train,y_train,eeta=lrate,threshold=1e-10)
        y_pred=predict(my_nn,X_test)
        print("accuracy :")
        print(getacc(y_pred,y_test))
        print()