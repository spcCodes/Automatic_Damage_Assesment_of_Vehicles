# import the necessary packages
from config import config
from imutils import paths
import shutil
import os
from sklearn.model_selection import train_test_split
import splitfolders

splitfolders.ratio('data/minor_moderate_severe', output="dataset/minor_moderate_severe", seed=1337, ratio=(.8, 0.1,0.1)) 


# # loop over the data splits
# for split in (config.TRAIN, config.TEST, config.VAL):

#     # grab all image paths in the current split
#     print("[INFO] processing '{} split'...".format(split))
#     p = os.path.sep.join([config.ORIG_INPUT_DATASET, split])
#     imagePaths = list(paths.list_images(p))
    
#     # loop over the image paths
#     for imagePath in imagePaths:
#         # extract class label from the filename
#         filename = imagePath.split("/")[-1]
#         print(filename)
#         label = imagePath.split("/")[-2]
#         print(label)
#         # construct the path to the output directory
#         dirPath = os.path.sep.join([config.BASE_PATH, split, label])
#         # if the output directory does not exist, create it
#         if not os.path.exists(dirPath):
#             os.makedirs(dirPath)
#         # construct the path to the output image file and copy it
#         p = os.path.sep.join([dirPath, filename])
#         shutil.copy2(imagePath, p)

