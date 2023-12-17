# MRI Classifiers
In this repository, our objective is to implement classical machine leaening classifiers such as Support Vector Machines (SVM), Random Forest (RF), and others and also more advanced classifiers like vision transformer and resnet18 on MRI images. The goal is to classify these images and determine whether the subject has Alzheimer's disease or not.
Also the final goal is to determine the affect of classical methods of augmentation like sharpening, bluring and random erasing and new methos like using diffusion model to do the the augmentaion and see the result of each on the mentioned classifiers.
# Requirements
Open CMD in Windows or Terminal in Linux. Then run the following command:
```bash
pip install -r requirements.txt
```

# Data augmentation using diffision models
## Training DDPMs
To train Denoising diffusion probabalistic models for each class you need to only run the following for each class:
```bash
python src/train_unconditional.py --class_MRI <class mri> --num_epochs 400
```
instead of `<class mri>` you can use one the following classes that we have:
* MildDemented
* ModerateDemented
* NonDemented
* VeryMildDemented
when you train an indivisual model for each class, the model will be in the diffiusion_models directory(all the four of the models are needed for data augmentation using DDPMs)
## Data generation using DDPMs
To accomplish this you just need to run the following:
```bash
python data_generation.py --number_of_images 5000
```
it will make a folder called generated in the data directory that has number_of_images of each class in it.
# Parameters
To modify the parameters you can open config.yaml in configs folder and modify it.
* Note: for data augmentation using classical methods you should do the following:
  + do_augmentation: true
  + do_generation: false
* and for data augmentation using DDPMs:
  + do_augmentation: true
  + do_generation: true

# Run
In the Terminal or CMD just run the following command have the results in the result folder:
```bash
python src/main.py
```

