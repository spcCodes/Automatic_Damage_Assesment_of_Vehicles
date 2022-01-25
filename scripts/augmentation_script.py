import albumentations as A
import cv2
import glob
import os,sys


folders=['minor_moderate_severe', 'car_damaged', 'front_rear_side']
sub_folders= ["train", "test" , "val"]

data_dir = "dataset/data"

transform = A.Compose([
    #A.Rotate(limit=(-20,20), p=0.5),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=25, p=0.5)
])


for folder in folders:
    for subf in sub_folders:
        path = os.path.join(data_dir , folder , subf)
        dir_list = os.listdir(path)
        for dir in dir_list:
            if dir ==".DS_Store":
                continue
            else:
                final_path = glob.glob(os.path.join(data_dir , folder , subf , dir) +"/*")
                
                #now for each image looping through
                for img in final_path:
                    print("Image is:",img)
                    image_name = img.replace("dataset/data" , "dataset/data_aug2").replace("." , "_augmented.")
                      # Read an image with OpenCV and convert it to the RGB colorspace
                    image = cv2.imread(img)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    # Augment an image
                    transformed = transform(image=image)
                    transformed_image = transformed["image"]
                    transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(image_name, transformed_image)

                

