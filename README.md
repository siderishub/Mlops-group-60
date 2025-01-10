# Mlops-group-60
An Mlops project for the 02476 course on an pneumonia dataset
Test

## Project Description

### Goal

The goal of the project is to use a dataset of chest X-ray images to create an image classification model for pneumonia detection. Our focus is to then use this model and project to further explore machine learning operation practices similar to those learned in the course and apply them to our project. By applying different models, we try to achieve a project that checks the best practices of machine learning operations and gives many opportunities to learn the practical application of the course content. For this reason, there will be less emphasis on the complexity of the model to make more room to work on the MLOps stack.

### Framework

The framework which that we chose to use to complete the project is the [PyTorch Image Models](https://github.com/rwightman/pytorch-image-models) from Hugging Face. This framework will be used to gain access to a plethora of pre-trained image models to apply to our data. To increase the performance for our specific dataset, we will fine tune the pre-trained model for our dataset. 


### Data

The dataset of [Chest X-ray Images](https://www.kaggle.com/datasets/tolgadincer/labeled-chest-xray-images) was found on Kaggle. It consists of 5,856 images of X-ray chest images. Around 4,000 images are of patients with pneumonia and around 1,500 are of patients who don't have pneumonia. For data preparation and augmentation we will establish a pipeline to clean and prepare the images for training. Further we add simple image augmentation techniques to improve the model generalization.


### Models
We are going to use models for image classification which are provided by the PyTorch Image Models library. There are many models to choose from, the one we are planning to usecurrently is [MobileNet V3](https://huggingface.co/timm/mobilenetv3_small_050.lamb_in1k), with the possibility of adding and comparing more models later on to see what works best and experiment. This model should provide a good starting point with a balance between efficiency and performance, which is helpful for quick developement and leaves time for experimentation.