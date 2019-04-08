import pickle

import cv2
import os,os.path
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def histogram_equalization(image):
    equ = cv2.equalizeHist(image)
    return equ


def noise_removal(image):
    dst = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    return dst


def grayScale_enhansment(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray


def pre_processing(image):
    image_nr = noise_removal(image)  # Noise Removal
    image_gray = grayScale_enhansment(image_nr)  # Gray scale
    image_heq = histogram_equalization(image_gray)  # Histogram Equalization
    return image_heq


def feature_extraction(image):
    surf = cv2.xfeatures2d.SURF_create()
    kp, des = surf.detectAndCompute(image, None)
    return des

def main():
    filename = 'finalized_model.sav'
    filename1 = 'testfeat.txt'
    filename2 = 'trainfeat.txt'
    filename3 = 'testlabel.txt'
    filename4 = 'trainlabel.txt'

    model = pickle.load(open(filename, 'rb'))
    testFeat = pickle.load(open(filename1, 'rb'))
    trainFeat = pickle.load(open(filename2, 'rb'))
    testLabels = pickle.load(open(filename3, 'rb'))
    trainLabels = pickle.load(open(filename4, 'rb'))

    acc = model.score(testFeat, testLabels)
    print(acc*100)



def train():
  imageDir = "images/"  # specify your path here
  image_path_list = []
  labels = []
  features = []
  for file in os.listdir(imageDir):
     image_path_list.append(os.path.join(imageDir, file))
  BOW = cv2.BOWKMeansTrainer(len(image_path_list))
  for imagePath in image_path_list:
     image = cv2.imread(imagePath)
     label = (imagePath.split(os.path.sep)[0].split(".")[0]).split("/")[1]
     image_app = pre_processing(image)
     keypoint_discriptor = feature_extraction(image_app)
     BOW.add(keypoint_discriptor)
     if label == "Skin_Infection":
         labels.append(0)
     else:
         labels.append(1)


  (trainFeat, testFeat, trainLabels, testLabels) = train_test_split(BOW.cluster(), labels, test_size=0.25, random_state=42)
  trainFeat= np.asarray(trainFeat)
  trainLabels = np.asarray(trainLabels)


  model = KNeighborsClassifier(n_neighbors=1,n_jobs=-1)
  model.fit(trainFeat, trainLabels)

  filename = 'finalized_model.sav'
  filename1 = 'testfeat.txt'
  filename2 = 'trainfeat.txt'
  filename3 = 'testlabel.txt'
  filename4 = 'trainlabel.txt'

  pickle.dump(model, open(filename, 'wb'))
  pickle.dump(testFeat, open(filename1, 'wb'))
  pickle.dump(trainFeat, open(filename2, 'wb'))
  pickle.dump(testLabels, open(filename3, 'wb'))
  pickle.dump(trainLabels, open(filename4, 'wb'))

def extratesting():
    imageDir = "extratest/"  # specify your path here
    image_path_list = []
    labels = []
    for file in os.listdir(imageDir):
        image_path_list.append(os.path.join(imageDir, file))
    BOW = cv2.BOWKMeansTrainer(len(image_path_list))
    for imagePath in image_path_list:
        image = cv2.imread(imagePath)
        label = (imagePath.split(os.path.sep)[0].split(".")[0]).split("/")[1]
        image_app = pre_processing(image)
        keypoint_discriptor = feature_extraction(image_app)
        BOW.add(keypoint_discriptor)
        if label == "Skin_Infection":
            labels.append(0)
        else:
            labels.append(1)
    print (BOW.cluster())
    test_data=BOW.cluster()
    test_data=np.asarray(test_data)
    filename = 'finalized_model.sav'
    model = pickle.load(open(filename, 'rb'))
    acc = model.score(test_data, labels)
    print(acc * 100)

main()