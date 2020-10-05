# Alphabet-Recognisation-Using-Hand-Gesture

## Overview</br>
This is a simple app to predict the alphabet that is written on screen using the blue coloured object ( you may change colour .... by changing the upper and lower limit of hsv of that particular colour . Screen act as if like a blackboard . Model of EMNIST is trained on the platform of Keras API .  </br>
</br>
Opencv is highly used in it to find the image processing and other such stuffs .</br>

![4hdhr3](https://user-images.githubusercontent.com/58811384/95072324-a39d1880-0728-11eb-9170-33855833d08b.gif)
## Motivation </br>
What could have been a perfect way to utilise the lockdown period ? Like most of my time in painting , Netflix . I thought of to start with the Deep Learning . After completing the Artificial Neural Network . I move on to CNN .So this is under that I made .

## Technical Aspects</br>
The Project is divided into three parts:
  1-> Make a model using EMNIST Alphabet dataset using Keras to predict the alphabet
  2-> Take the reference of blue colour to draw on the screen . and using a deque to store the point of location where the blue object ( reference ) is moving . and predict the alphabet 
  3-> Adding a feature of sound ( to speak the predicted alphabet) . 
  

## Installation </br>
The code is written in python 3.7 . It you don't have installed you can find it on google . If you have a lower version of Python you can upgrade using the pip package , ensuring you have the latest version of pip . To install the required Packages and libraries , run teh command in the project directory after cloning the repository. </br>

### pip install -r requirements.txt
</br>

 ## Running the code </br>
 After following the above steps of installation . Open the terminal( cmd, powershell ) in the project directory and use the command </br> 
 ### python GestureUsingCap.py
 </br>


## Main Libraries required-
Numpy ( for n-dimension array )</br>
PIL ( for image manipulation )</br>
Keras ( To train the model )</br>
Tensorflow-gpu ( Google API for deep learning )</br>
Tensorboard ( Visual analysis os model )</br>
Opencv-python ( Scientific library for Image Related stuffs )</br>
