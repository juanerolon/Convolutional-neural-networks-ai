#Scratch work
from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
from glob import glob

# define function to load train, test, and validation datasets
######################################################################
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets


mpath = '/mnt/linuxdata2/Dropbox/_machine_learning/udacity_projects/cnn-project/'


#------------------------------- Load  dog images dataset ---------------------------------------------

# load train, test, and validation datasets
train_files, train_targets = load_dataset(mpath+'dogImages/train')
valid_files, valid_targets = load_dataset(mpath+'dogImages/valid')
test_files, test_targets = load_dataset(mpath+'dogImages/test')

# load list of dog names
dog_names = [item[20:-1] for item in sorted(glob(mpath+"dogImages/train/*/"))]

# print statistics about the dataset
print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.'% len(test_files))

#------------------------------ Load human faces dataset -------------------------------------------------

import random
random.seed(8675309)

# load filenames in shuffled human dataset
human_files = np.array(glob(mpath+"lfw/*/*"))
random.shuffle(human_files)

# print statistics about the dataset
print('There are %d total human images.' % len(human_files))

# ------------------------------ Detect human faces in images ---------------------------------------------

import cv2
import matplotlib.pyplot as plt
#%matplotlib inline

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier(mpath+'haarcascades/haarcascade_frontalface_alt.xml')

# load color (BGR) image
img = cv2.imread(human_files[3])
# convert BGR image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find faces in image
faces = face_cascade.detectMultiScale(gray)

# print number of faces detected in the image
print('Number of faces detected:', len(faces))

# get bounding box for each detected face
for (x,y,w,h) in faces:
    # add bounding box to color image
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

# convert BGR image to RGB for plotting
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# display the image, along with bounding box
plt.imshow(cv_rgb)
plt.show()

#
# returns "True" if face is detected in image stored at img_path
#########################################################################
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0