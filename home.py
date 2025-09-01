import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import cv2
import numpy as np

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import math
import csv

import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from model.mnist_model import MNIST_CNN

# MNIST class
class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()

        # define input, hidden, output
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)  # 1 channel in, 10 out
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)  # 10 channels in, 20 out
        self.conv2_dropout = nn.Dropout2d()  # dropout layer is a regualarization layer (randomly deactivates certain network nodes)
        self.fcl1 = nn.Linear(320, 50)
        self.fcl2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_dropout(self.conv2(x)), 2))

        # flatten data
        x = x.view(-1, 320)
        x = F.relu(self.fcl1(x))
        x = F.dropout(x, training=self.training)
        x = self.fcl2(x)

        return x


# first make the drawing website then add functionalities, then make it modular (with classes and stuff)
class WebsiteBuild():
    # build website from top down
    def __init__(self):
        st.set_page_config("MNIST Classifier", layout="centered")


    def image_to_tensor(self, img):
        #resize image to MNIST format
        final_img = cv2.resize(img, (28,28), interpolation=cv2.INTER_AREA)
        # img = 255 - img  # invert the colors

        # turn to a tensor
        data = torch.from_numpy(final_img).unsqueeze(0).float() / 255.0
        # print(data.min().item(), data.max().item())

        data = torch.reshape(data, (1, 28, 28))

        # print("TENSOR SHAPE:" , data.shape)
        
        return data
    

    def predict_digit(self, img_array):
        img_tensor = self.image_to_tensor(img_array)
        print(img_tensor.shape)

        # call the PyTorch model and run the predict method
        model = MNIST_CNN()
        model.load_state_dict(torch.load("./model/MNIST_CNN_model.pth", map_location="cpu"))
        model.eval()       

        # prediction
        output = model(img_tensor) 
        prediction = output.argmax(dim=1, keepdim=True).item()

        # confidence score
        probs = F.softmax(output, dim=1)
        confidence = probs.max().item()

        print(f"Prediction: {prediction}")
        print(f"Confidence: {confidence*100:.2f}")
        
        return img_tensor, prediction, confidence


    def homepage(self):
        st.title("MNIST Digit Classifier")
        st.write("Draw a digit (0-9), press the 'Predict' button and the AI will try to guess what you wrote.")
        
        #canvas
        canvas_result = st_canvas(
            fill_color="rgba(0,0,0,1)",
            stroke_width=10,
            stroke_color="white",
            background_color="black",
            width=280,
            height=280,
            drawing_mode="freedraw",
            key="canvas"   
        )

        predict_btn = st.button("Predict")
        clear_btn = st.button("Clear")

        # TODO: clear button functionality

        # convert the canvas into an image for the model
        if predict_btn and canvas_result.image_data is not None:
            img = canvas_result.image_data.astype(np.uint8)
            grey_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img_val, prediction, confidence = self.predict_digit(grey_img)

            col1, col2 = st.columns(2)
            col1.image(grey_img, caption="Processed Input", width=100)
            col2.write(f"Image turned into Tensor with size: {img_val.shape}")
            # col2.write(f"Image turned into tensor: {img_val[0, :5, :5]}")

            st.write(f"Model Prediction: {prediction}")
            st.write(f"Model Confidence: {confidence*100:.2f}%")

            # add the img, prediction, confidence into a dataset
            # with open("data.csv", "a+", newline="") as model_data:
            #     mnist_data = csv.writer(model_data, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)
            #     mnist_data.writerow([img_val, prediction, confidence])




def main():
    web = WebsiteBuild()
    web.homepage()


if __name__ == "__main__":
    main()