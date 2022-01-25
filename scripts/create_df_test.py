import os , sys, glob
import pandas as pd


car_damage = ['damage' , 'not_damaged']
final_result =[]
for idx,i in enumerate(car_damage):
  files = glob.glob(os.path.join('/content/drive/MyDrive/damage_assess_sahaj/dataset/data/car_damaged/test' , i)+'/*')
  for file in files:
    result={}
    result["name"] = file
    result["gt"] = idx
    final_result.append(result)


df = pd.DataFrame(final_result)
df.to_csv(os.path.join('/content/drive/MyDrive/damage_assess_sahaj/dataset/data/car_damaged/', 'car_damage_test.csv'))

frs = ['front' , 'rear' , 'side']
final_result =[]
for idx,i in enumerate(frs):
  files = glob.glob(os.path.join('/content/drive/MyDrive/damage_assess_sahaj/dataset/data/front_rear_side/test' , i)+'/*')
  for file in files:
    result={}
    result["name"] = file
    result["gt"] = idx
    final_result.append(result)


df = pd.DataFrame(final_result)
df.to_csv(os.path.join('/content/drive/MyDrive/damage_assess_sahaj/dataset/data/front_rear_side', 'frs_test.csv'))


intensity = ['minor' , 'moderate' , 'severe']
final_result =[]
for idx,i in enumerate(intensity):
  files = glob.glob(os.path.join('/content/drive/MyDrive/damage_assess_sahaj/dataset/data/minor_moderate_severe/test' , i)+'/*')
  for file in files:
    result={}
    result["name"] = file
    result["gt"] = idx
    final_result.append(result)


df = pd.DataFrame(final_result)
df.to_csv(os.path.join('/content/drive/MyDrive/damage_assess_sahaj/dataset/data/minor_moderate_severe', 'intensity_test.csv'))

