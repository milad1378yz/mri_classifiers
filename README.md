# MRI Classifiers

## Introduction
This repository is dedicated to implementing and comparing various machine learning classifiers on MRI images for Alzheimer's disease detection. Our approach includes classical classifiers like Support Vector Machines (SVM) and Random Forest (RF), as well as advanced models such as vision transformers and ResNet-18. We also explore the impact of traditional and novel data augmentation methods, including sharpening, blurring, random erasing, and the use of diffusion models, on classifier performance.

## Requirements
To set up the environment, open CMD in Windows or Terminal in Linux, and execute the following command:
```bash
pip install -r requirements.txt
```
Also for data files you need to install 'git LFS'.

## Data Augmentation Using Diffusion Models

### Training Denoising Diffusion Probabilistic Models (DDPMs)
To train DDPMs for each MRI class, execute the command below for each class:
```bash
python src/train_unconditional.py --class_MRI <class mri> --num_epochs 400
```
Replace `<class mri>` with one of the following classes:
- MildDemented
- ModerateDemented
- NonDemented
- VeryMildDemented

Training individual models for each class stores them in the `diffusion_models` directory. All four models are required for data augmentation using DDPMs.

### Generating Data Using DDPMs
To generate augmented data:
```bash
python src/data_generation.py --number_of_images 5000
```
This creates a `generated` folder in the `data` directory, containing 5000 images for each class.

## Configurations
Configuration adjustments can be made in the `config.yaml` file in the `configs` folder. For different augmentation methods:
- Classical methods:
  + `do_augmentation: true`
  + `do_generation: false`
- Using DDPMs:
  + `do_augmentation: true`
  + `do_generation: true`

## Execution
To run the project and store results in the `result` folder, use:
```bash
python src/main.py
```


## Docker
If you want to use docker automatically:
* to build: 
```bash
cd docker
docker build -t env_image .
```
* to run: 
```bash
cd ..
docker run -d -it --name container_runner -v .:/app env_image tail -f /dev/null
docker exec -it container_runner bash
pip install -r requirements.txt
```
Now in the docker container you can do the following things:

- The rest is the same as above.