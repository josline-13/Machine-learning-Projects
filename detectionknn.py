import numpy as np
import cv2

cap = cv2.VideoCapture(0)

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default')

person01 = np.load('JC.npy').reshape(20,50*50*3)
person02 = np.load('AJ.npy').reshape(20,50*50*3)
#person03 = np.load('Naimisha.npy').reshape(20,50*50*3)

names = {
        0:'Josline',
        1:'Adriel',
#        2:'Naimisha'
        }

data = np.concatenate([person01,person02])
labels = np.zeros((40,1))
labels[:21,:] = 0.0
labels[21:,:] = 1.0

def distance(x1,x2):
    return np.sqrt((x1-x2)**2).sum()


def knn(testinput, data,labels,k):
    numRows = data.shape[0]
    dist = []
    for i in range(m):
        distance(x,data[i])
