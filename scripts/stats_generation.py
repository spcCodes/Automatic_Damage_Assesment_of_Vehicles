import os, sys
import glob
import matplotlib.pyplot as plt
import seaborn as sns

def plot_bar(result,name, save_fig):
    plt.figure(figsize = (5,5))
    f = sns.barplot(x = list(result.keys()),y =list(result.values()))
    plt.xlabel("Class labels", fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title("Number of Images in "+name+' folder', fontsize=15)
    plt.savefig(save_fig)
    

folders=['minor_moderate_severe', 'car_damaged', 'front_rear_side']
sub_folders= ["train", "test" , "val"]
data_dir = "dataset/data_aug2"

if not os.path.exists(os.path.join(data_dir, "figures")):
    os.makedirs(os.path.join(data_dir, "figures"))

for folder in folders:
    for subf in sub_folders:
        path = os.path.join(data_dir , folder , subf)
        dir_list = os.listdir(path)
        dir_list.remove(".DS_Store")
        result={}
        for dir in dir_list:
            final_path = glob.glob(os.path.join(data_dir , folder , subf , dir) +"/*")
            result[dir] = len(final_path)
            figure_name = folder+"_"+subf +".jpg"
            plot_bar(result,subf, os.path.join(data_dir, "figures",figure_name))
            print("For "+ folder + ' , Number of '+dir+' images in ' + subf + " is " +str(len(final_path)))
