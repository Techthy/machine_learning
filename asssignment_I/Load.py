import numpy as np
from imutils import paths
from PIL import Image as im


def load_training_set():

    

    imagepathsA = list(paths.list_images("data/train/A"))
    imagepathsB = list(paths.list_images("data/train/B"))
    imagepathsC = list(paths.list_images("data/train/C"))

    imagesA2 = []
    imagesB2 = []
    imagesC2 = []

    for i in range(0,2000):
        imagesA2.append(np.array((im.open(imagepathsA[i]))).flatten())
        imagesB2.append(np.array((im.open(imagepathsB[i]))).flatten())
        imagesC2.append(np.array((im.open(imagepathsC[i]))).flatten())


    imagesA = np.array(imagesA2)
    YA = np.ones([2000,1])

    imagesB = np.array(imagesB2)
    YB = -np.ones([2000,1])

    imagesC = np.array(imagesC2)
    YC = -np.ones([2000,1])

    X = np.concatenate((imagesA, imagesB),axis=0)
    X = np.concatenate((X,imagesC),axis=0)
    Y = np.concatenate((YA,YB), axis=0)
    Y = np.concatenate((Y, YC), axis=0)

    return X, Y

def load_test_set():
    imagepathsA = list(paths.list_images("data1/data/test/A"))
    imagepathsB = list(paths.list_images("data1/data/test/B"))
    imagepathsC = list(paths.list_images("data1/data/test/C"))

    imagesA2 = []
    imagesB2 = []
    imagesC2 = []

    for i in range(0,250):
        imagesA2.append(np.array((im.open(imagepathsA[i]))).flatten())
        imagesB2.append(np.array((im.open(imagepathsB[i]))).flatten())
        imagesC2.append(np.array((im.open(imagepathsC[i]))).flatten())


    imagesA = np.array(imagesA2)
    YA = np.ones([250,1])

    imagesB = np.array(imagesB2)
    YB = -np.ones([250,1])

    imagesC = np.array(imagesC2)
    YC = -np.ones([250,1])

    X = np.concatenate((imagesA, imagesB),axis=0)
    X = np.concatenate((X,imagesC),axis=0)
    Y = np.concatenate((YA,YB), axis=0)
    Y = np.concatenate((Y, YC), axis=0)

    return X, Y