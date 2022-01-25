######## main function ############
from fastai.vision import *
from fastai.metrics import error_rate, accuracy
import warnings
warnings.filterwarnings('ignore')
from google.colab import drive
drive.mount('/content/drive')
import matplotlib.pyplot as plt
from fastai.callbacks import *
import os , sys

# Set path to root directory
path = Path('/content/drive/MyDrive/damage_assess_sahaj/dataset/data/car_damaged')


def clear_pyplot_memory():
    plt.clf()
    plt.cla()
    plt.close()

data_dir = "/content/drive/MyDrive/damage_assess_sahaj/dataset/data/minor_moderate_severe"

# Set path to root directory
path = Path(data_dir)
data = ImageDataBunch.from_folder(path, train='train', valid='val', ds_tfms=get_transforms(do_flip=False), size=224, bs=64, num_workers=8)
#show some images
data.show_batch()
plt.savefig(os.path.join(data_dir,'data_batch.png'))
clear_pyplot_memory()


classes = data.classes
num_classes = data.c

model_name = ["vgg16_bn" , "vgg19_bn" , "resnet50" , "densenet201"]

for model_i in model_name:

  if model_i == "vgg16_bn":
    # Build the CNN model with the pretrained resnet34
    learn = cnn_learner(data, models.vgg16_bn, metrics = [accuracy, FBeta(), Precision(), Recall()],callback_fns=[CSVLogger])
    # Train the model on 4 epochs of data at the default learning rate
    learn.fit_one_cycle(4)
    learn.csv_logger.read_logged_file()

    os.rename(os.path.join(data_dir,'history.csv'),os.path.join(data_dir,model_i+"_"+'fc_training.csv')) 

    # Save the model
    learn.save(model_i+ "_"+ 'stage-1')

    # Unfreeze all layers of the CNN
    learn.unfreeze()
    
    #full layer training
    learn.fit_one_cycle(2, max_lr=slice(3e-7, 3e-6))
    learn.fit_one_cycle(4, max_lr=slice(1e-4, 1e-3))
    learn.csv_logger.read_logged_file()

    os.rename(os.path.join(data_dir,'history.csv'),os.path.join(data_dir,model_i+"_"+'full_training.csv'))

    # Rebuild interpreter and replot confusion matrix
    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
    plt.savefig(os.path.join(data_dir,model_i+"_"+'cm_val.png'))
    clear_pyplot_memory()

    # Save the model
    learn.save(model_i+ "_" + 'stage-2')

    interp.plot_top_losses(9, figsize=(15,11))
    plt.savefig(os.path.join(data_dir,model_i+"_"+'top_losses.png'))
    clear_pyplot_memory()

    #exporting the file
    learn.export()
    os.rename(os.path.join(data_dir,'export.pkl'),os.path.join(data_dir,model_i+"_"+'export.pkl'))
    
  elif model_i == "vgg19_bn":

    # Build the CNN model with the pretrained resnet34
    learn = cnn_learner(data, models.vgg19_bn, metrics = [accuracy, FBeta(), Precision(), Recall()],callback_fns=[CSVLogger])
    # Train the model on 4 epochs of data at the default learning rate
    learn.fit_one_cycle(4)
    learn.csv_logger.read_logged_file()

    os.rename(os.path.join(data_dir,'history.csv'),os.path.join(data_dir,model_i+"_"+'fc_training.csv')) 

    # Save the model
    learn.save(model_i+ "_"+ 'stage-1')

    # Unfreeze all layers of the CNN
    learn.unfreeze()
    
    #full layer training
    learn.fit_one_cycle(2, max_lr=slice(3e-7, 3e-6))
    learn.fit_one_cycle(4, max_lr=slice(1e-4, 1e-3))
    learn.csv_logger.read_logged_file()

    os.rename(os.path.join(data_dir,'history.csv'),os.path.join(data_dir,model_i+"_"+'full_training.csv'))

    # Rebuild interpreter and replot confusion matrix
    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
    plt.savefig(os.path.join(data_dir,model_i+"_"+'cm_val.png'))
    clear_pyplot_memory()

    # Save the model
    learn.save(model_i+ "_" + 'stage-2')

    interp.plot_top_losses(9, figsize=(15,11))
    plt.savefig(os.path.join(data_dir,model_i+"_"+'top_losses.png'))
    clear_pyplot_memory()

    #exporting the file
    learn.export()
    os.rename(os.path.join(data_dir,'export.pkl'),os.path.join(data_dir,model_i+"_"+'export.pkl'))

  elif model_i == "resnet50":
    # Build the CNN model with the pretrained resnet34
    learn = cnn_learner(data, models.resnet50, metrics = [accuracy, FBeta(), Precision(), Recall()],callback_fns=[CSVLogger])
    # Train the model on 4 epochs of data at the default learning rate
    learn.fit_one_cycle(4)
    learn.csv_logger.read_logged_file()

    os.rename(os.path.join(data_dir,'history.csv'),os.path.join(data_dir,model_i+"_"+'fc_training.csv')) 

    # Save the model
    learn.save(model_i+ "_"+ 'stage-1')

    # Unfreeze all layers of the CNN
    learn.unfreeze()
    
    #full layer training
    learn.fit_one_cycle(2, max_lr=slice(3e-7, 3e-6))
    learn.fit_one_cycle(4, max_lr=slice(1e-4, 1e-3))
    learn.csv_logger.read_logged_file()

    os.rename(os.path.join(data_dir,'history.csv'),os.path.join(data_dir,model_i+"_"+'full_training.csv'))

    # Rebuild interpreter and replot confusion matrix
    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
    plt.savefig(os.path.join(data_dir,model_i+"_"+'cm_val.png'))
    clear_pyplot_memory()

    # Save the model
    learn.save(model_i+ "_" + 'stage-2')

    interp.plot_top_losses(9, figsize=(15,11))
    plt.savefig(os.path.join(data_dir,model_i+"_"+'top_losses.png'))
    clear_pyplot_memory()

    #exporting the file
    learn.export()
    os.rename(os.path.join(data_dir,'export.pkl'),os.path.join(data_dir,model_i+"_"+'export.pkl'))

  elif model_i == "densenet201":
    # Build the CNN model with the pretrained resnet34
    learn = cnn_learner(data, models.densenet201, metrics = [accuracy, FBeta(), Precision(), Recall()],callback_fns=[CSVLogger])
    # Train the model on 4 epochs of data at the default learning rate
    learn.fit_one_cycle(4)
    learn.csv_logger.read_logged_file()

    os.rename(os.path.join(data_dir,'history.csv'),os.path.join(data_dir,model_i+"_"+'fc_training.csv')) 

    # Save the model
    learn.save(model_i+ "_"+ 'stage-1')

    # Unfreeze all layers of the CNN
    learn.unfreeze()
    
    #full layer training
    learn.fit_one_cycle(2, max_lr=slice(3e-7, 3e-6))
    learn.fit_one_cycle(6, max_lr=slice(1e-4, 1e-3))
    learn.csv_logger.read_logged_file()

    os.rename(os.path.join(data_dir,'history.csv'),os.path.join(data_dir,model_i+"_"+'full_training.csv'))

    # Rebuild interpreter and replot confusion matrix
    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
    plt.savefig(os.path.join(data_dir,model_i+"_"+'cm_val.png'))
    clear_pyplot_memory()

    # Save the model
    learn.save(model_i+ "_" + 'stage-2')

    interp.plot_top_losses(9, figsize=(15,11))
    plt.savefig(os.path.join(data_dir,model_i+"_"+'top_losses.png'))
    clear_pyplot_memory()

    #exporting the file
    learn.export()
    os.rename(os.path.join(data_dir,'export.pkl'),os.path.join(data_dir,model_i+"_"+'export.pkl'))
  
