# Homework 11

# K-Means from Scratch

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import matplotlib.pyplot as plt
import keras
from tqdm import tqdm
import click

import numpy as np

class cluster():
    def __init__(self,epochs,kn,maxiter,train,test):
        self.kn = kn
        self.epoch = epochs
        self.tl = train
        self.ttl = test
        self.maxiter = maxiter
        self.xtrain, self.ytrain, self.xtest,self.ytest = self.data_prep()
        self.centers = self.train()

    def data_prep(self):
        assert self.tl <= 60000
        assert self.ttl <= 10000
        (X1,Y1), (X2,Y2) = keras.datasets.mnist.load_data()
        xtrain = np.divide(np.array(X1.reshape(60000,784)),255)
        xtrain = xtrain[:self.tl,:]
        ytrain = np.array(Y1[:self.tl])
        xtest = np.divide(np.array(X2.reshape(10000,784)),255)
        xtest = xtest[:self.ttl,:]
        ytest = np.array(Y2[:self.ttl])
        assert xtrain.ndim == 2
        assert ytrain.ndim == 1
        assert xtest.ndim == 2
        assert ytest.ndim == 1

        return xtrain,ytrain,xtest,ytest
    
    def fit_step(self,data,centers):
        data1 = data
        pred = np.zeros(len(data1))
        dist = np.linalg.norm(data1[:, None, :] - centers[None, :, :],axis = 2)
        pred = np.argmin(dist, axis = 1)

        c = np.arange(self.kn)
        mask = pred == c[:,None]
        sums = np.where(mask[:, :, None], data1, 0).sum(axis = 1)
        counts = np.count_nonzero(mask, axis = 1).reshape(self.kn,1)
        centers = sums/counts

        return centers
    
    def label_step(self,data,centers):
        label = {}
        dist = np.linalg.norm(data[:, None, :] - centers[None, :, :],axis = 2)
        pred = np.argmin(dist, axis = 1)
        for i in range(self.kn):
            index = np.where(pred == i)
            num = np.bincount(self.ytrain[index]).argmax()
            label[i] = num.item()
        return label
    
    def pred_step(self,data,centers):
        label = self.label_step(data,centers)
        dist = np.linalg.norm(data[:, None, :] - centers[None, :, :],axis = 2)
        pred = np.argmin(dist, axis = 1)
        res = np.zeros(len(pred))
        for i in range(len(pred)):
            z = pred[i].item()
            res[i] = label.get(z)
        return res

    def batch_fit(self,pred,centers):
        for i in tqdm(range(self.maxiter)):
            new_centers = self.fit_step(self.xtrain,centers)
            new_pred = self.pred_step(self.xtrain,new_centers)

            if np.all(new_pred == pred):
                break           
            pred = new_pred
            centers = new_centers

        return pred,centers
    
    def train(self):
        pred = np.zeros(self.tl)
        initial_center = np.random.choice(self.tl,self.kn,replace=False)
        centers = self.xtrain[initial_center]
        for i in tqdm(range(self.epoch)):
            pred,ncenters = self.batch_fit(pred,centers)
            centers = ncenters
        
        testpred = self.test_step(self.xtest,centers)

        accuracy = np.divide(np.count_nonzero(testpred == self.ytest),len(testpred))
        print(accuracy)

        centers = np.multiply(centers,255)
        print(np.max(centers))

        fig, bx1 = plt.subplots(2,5)
        c0,c1,c2,c3,c4,c5,c6,c7,c8,c9 = np.array_split(centers,10,axis = 0)
        c0 = np.reshape(c0,(28,28))
        c1 = np.reshape(c1,(28,28))
        c2 = np.reshape(c2,(28,28))
        c3 = np.reshape(c3,(28,28))
        c4 = np.reshape(c4,(28,28))
        c5 = np.reshape(c5,(28,28))
        c6 = np.reshape(c6,(28,28))
        c7 = np.reshape(c7,(28,28))
        c8 = np.reshape(c8,(28,28))
        c9 = np.reshape(c9,(28,28))
        bx1[0,0].matshow(c0,cmap='gray', vmin=0, vmax=255)
        bx1[0,1].matshow(c1,cmap='gray', vmin=0, vmax=255)
        bx1[0,2].matshow(c2,cmap='gray', vmin=0, vmax=255)
        bx1[0,3].matshow(c3,cmap='gray', vmin=0, vmax=255)
        bx1[0,4].matshow(c4,cmap='gray', vmin=0, vmax=255)
        bx1[1,0].matshow(c5,cmap='gray', vmin=0, vmax=255)
        bx1[1,1].matshow(c6,cmap='gray', vmin=0, vmax=255)
        bx1[1,2].matshow(c7,cmap='gray', vmin=0, vmax=255)
        bx1[1,3].matshow(c8,cmap='gray', vmin=0, vmax=255)
        bx1[1,4].matshow(c9,cmap='gray', vmin=0, vmax=255)
        fig.savefig('centroids.pdf')

        return centers,testpred
    
    def test_step(self,data,centers):
        label = self.label_step(data,centers)
        dist = np.linalg.norm(data[:, None, :] - centers[None, :, :],ord=3,axis = 2)
        pred = np.argmin(dist, axis = 1)
        res = np.zeros(len(data))
        for i in range(len(data)):
            z = pred[i].item()
        res[i] = label.get(z)
        return res

# Optional Command Line Flags, Default is standard values run (all mnist records, 10 clusters, 5 epochs, 10 iterations per epoch)
@click.command()
@click.option(
    '--epochs','-e',
    default=5,
    show_default=True,
    help='Max Iterations'
)
@click.option(
    '--kn',
    default=10,
    show_default=True,
    help='Max Iterations'
)
@click.option(
    '--maxiter','-m',
    default=10,
    show_default=True,
    help='Max Iterations'
)
@click.option(
    '--train','-t',
    default=60000,
    show_default=True,
    help='Number of Training Items'
)
@click.option(
    '--test','-tt',
    default=10000,
    show_default=True,
    help='Number of Testing Items'
)

def main(epochs,kn,maxiter,train,test):
    c1 = cluster(epochs,kn,maxiter,train,test)

if __name__ == "__main__":
    plt.ion()
    main()