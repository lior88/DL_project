# DL project - Traffic sign recognition
<h1 align="center">
  <br>
Technion EE 046211 - Deep Learning
  <br>
  <img src="https://raw.githubusercontent.com/taldatech/ee046211-deep-learning/main/assets/nn_gumgum.gif" height="200">
</h1>
  <p align="center">
    <a href="https://github.com/lior88">Li-or Bar David </a> •
    <a> Kfir Levi </a> •
  </p>


- [DL project - Traffic sign recognition](#dl-project---traffic-sign-recognition)
  * [Agenda](#agenda)
  * [Running The Code](#running-the-code)
  * [The GTSRB dataset](#the-gtsrb-dataset)
  * [References](#references)


## Agenda

|File       | Topics Covered |
|----------------|---------|
|`main.py`| the main file of the project |
|`Networks.py`| the implementations of the Original network and the Cifar_CNN network |


## Running The Code
In order to run the code, use the main.py file, it requires 3 parameters:
* TrainTest - receives string "Test" or "Train". use it to choose whether to train a model or test it.
* Classifier - receives a string which specifies the wanted Classifier. options are: 
  + Original - the original network used for the project.
  + CifarCNN - the Cifar_CNN network we saw in class.
  + resnet18 - the pretrained resnet18 network.
  + vgg16 - the pretrained vgg16 network.
* root - the location of the data folders.

## The GTSRB dataset
we used the GTSRB dataset, which consists of german traffic signs with 43 different classes and more than 39,000 samples of 30x30 RGB images.
in order to run the code, you need to download the dataset from one of the following sources:
  * 'https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign'
  * 'https://benchmark.ini.rub.de/gtsrb_dataset.html'


## References
the original project, which we used as a base can be found at:
  * https://data-flair.training/blogs/python-project-traffic-signs-recognition/
