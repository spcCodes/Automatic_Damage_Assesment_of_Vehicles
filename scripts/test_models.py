from PIL import Image as PImage
from fastai.vision import *
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os , sys, glob
import pandas as pd


def clear_pyplot_memory():
    plt.clf()
    plt.cla()
    plt.close()


path = Path('/content/drive/MyDrive/damage_assess_sahaj/dataset/data_aug1/car_damaged')
path1 = Path('/content/drive/MyDrive/damage_assess_sahaj/dataset/data_aug1/front_rear_side')
path2 = Path('/content/drive/MyDrive/damage_assess_sahaj/dataset/data_aug1/minor_moderate_severe')

from fastai.vision import Path,load_learner,Image

learn_damaged_notdamaged = load_learner(path , 'resnet50_export.pkl')
learn_frs = load_learner(path1,'vgg16_bn_export.pkl')
learn_di = load_learner(path2,'densenet201_export.pkl')

df_damaged = pd.read_csv('/content/drive/MyDrive/damage_assess_sahaj/dataset/data/car_damaged/car_damage_test.csv')
y_pred =[]
for idx , row in df_damaged.iterrows():
  file_name = row["name"]
  p = cv2.imread(file_name) # p is numpy array with shape (height,width,channels)
  t = pil2tensor(p, dtype=np.uint8) # converts to numpy tensor
  t = t.float()/255. #Convert to float
  im = Image(t) # Convert to fastAi Image - this class has "apply_tfms"
  pred = learn_damaged_notdamaged.predict(im)
  y_pred.append(int(pred[1]))
y_gt = list(df_damaged["gt"])
import matplotlib.pyplot as plt

def clear_pyplot_memory():
    plt.clf()
    plt.cla()
    plt.close()
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
print(classification_report(y_gt, y_pred))
print(confusion_matrix(y_gt, y_pred))
confusion_matrix_df = pd.DataFrame(confusion_matrix(y_gt, y_pred)).rename(columns={0:'damaged',1:'not_damaged'}, index={0:'damaged',1:'not_damaged'})
fig, ax = plt.subplots(figsize=(7,5))         
sns.heatmap(confusion_matrix_df, annot=True, ax=ax,fmt='g')
plt.savefig(os.path.join('/content/drive/MyDrive/damage_assess_sahaj/dataset/data/car_damaged','damage'+"_"+'cm_test.png'))
clear_pyplot_memory()


df_frs = pd.read_csv('/content/drive/MyDrive/damage_assess_sahaj/dataset/data/front_rear_side/frs_test.csv')
y_pred =[]
for idx , row in df_frs.iterrows():
  file_name = row["name"]
  p = cv2.imread(file_name) # p is numpy array with shape (height,width,channels)
  t = pil2tensor(p, dtype=np.uint8) # converts to numpy tensor
  t = t.float()/255. #Convert to float
  im = Image(t) # Convert to fastAi Image - this class has "apply_tfms"
  pred = learn_frs.predict(im)
  y_pred.append(int(pred[1]))
y_gt = list(df_frs["gt"])
import matplotlib.pyplot as plt

def clear_pyplot_memory():
    plt.clf()
    plt.cla()
    plt.close()
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
print(classification_report(y_gt, y_pred))
print(confusion_matrix(y_gt, y_pred))
confusion_matrix_df = pd.DataFrame(confusion_matrix(y_gt, y_pred)).rename(columns={0:'front',1:'rear' , 2: 'side'}, index={0:'front',1:'rear' , 2: 'side'})
fig, ax = plt.subplots(figsize=(7,5))         
sns.heatmap(confusion_matrix_df, annot=True, ax=ax,fmt='g')
plt.savefig(os.path.join('/content/drive/MyDrive/damage_assess_sahaj/dataset/data/front_rear_side','frs'+"_"+'cm_test.png'))
clear_pyplot_memory()

df_intensity = pd.read_csv('/content/drive/MyDrive/damage_assess_sahaj/dataset/data/minor_moderate_severe/intensity_test.csv')
y_pred =[]
for idx , row in df_intensity.iterrows():
  file_name = row["name"]
  p = cv2.imread(file_name) # p is numpy array with shape (height,width,channels)
  t = pil2tensor(p, dtype=np.uint8) # converts to numpy tensor
  t = t.float()/255. #Convert to float
  im = Image(t) # Convert to fastAi Image - this class has "apply_tfms"
  pred = learn_di.predict(im)
  y_pred.append(int(pred[1]))
y_gt = list(df_intensity["gt"])
import matplotlib.pyplot as plt

def clear_pyplot_memory():
    plt.clf()
    plt.cla()
    plt.close()
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
print(classification_report(y_gt, y_pred))
print(confusion_matrix(y_gt, y_pred))
confusion_matrix_df = pd.DataFrame(confusion_matrix(y_gt, y_pred)).rename(columns={0:'minor',1:'moderate' , 2: 'severe'}, index={0:'minor',1:'moderate' , 2: 'severe'})
fig, ax = plt.subplots(figsize=(7,5))         
sns.heatmap(confusion_matrix_df, annot=True, ax=ax,fmt='g')
plt.savefig(os.path.join('/content/drive/MyDrive/damage_assess_sahaj/dataset/data/minor_moderate_severe','intensity'+"_"+'cm_test.png'))
clear_pyplot_memory()