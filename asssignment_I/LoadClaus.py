import os
from PIL import Image
import numpy as np

def import_images(directory):
    images = []
    valid_extensions = ['.png' ]  # Add more extensions if needed

    for filename in os.listdir(directory):
        if filename.lower().endswith(tuple(valid_extensions)):
            image_path = os.path.join(directory, filename)
            image = Image.open(image_path)
            images.append(np.array(image).flatten() / 255.0) # Normalize the images and convert to a vector (1D array)

    return np.array(images)



def load_training_dataset():
    script_dir = os.path.dirname(__file__)  # get the directory of the current script
    pathToTrainingDataA = os.path.join(script_dir, 'data/train/A')
    pathToTrainingDataB = os.path.join(script_dir, 'data/train/B')
    pathToTrainingDataC = os.path.join(script_dir, 'data/train/C')

    imagesA = import_images(pathToTrainingDataA)
    imagesB = import_images(pathToTrainingDataB)
    imagesC = import_images(pathToTrainingDataC)


    
    YA = np.ones([2000,1])
    YB = np.zeros([2000,1])
    YC = np.zeros([2000,1])

    X = np.concatenate((imagesA, imagesB, imagesC),axis=0)
    Y = np.concatenate((YA, YB, YC), axis=0)

    return X, Y


def load_test_dataset():


    script_dir = os.path.dirname(__file__)  # get the directory of the current script
    pathToTestDataA = os.path.join(script_dir, 'data/test/A')
    pathToTestDataB = os.path.join(script_dir, 'data/test/B')
    pathToTestDataC = os.path.join(script_dir, 'data/test/C')

    imagesA = import_images(pathToTestDataA)
    imagesB = import_images(pathToTestDataB)
    imagesC = import_images(pathToTestDataC)

    YA = np.ones([250,1])
    YB = np.zeros([250,1])
    YC = np.zeros([250,1])

    X = np.concatenate((imagesA, imagesB, imagesC),axis=0)
    Y = np.concatenate((YA,YB, YC), axis=0)

    return X, Y

