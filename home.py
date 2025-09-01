import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import cv2
import numpy as np
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import math
import torch

# first make the drawing website then add functionalities, then make it modular (with classes and stuff)

class WebsiteBuild():
    # build website from top down
    def __init__(self):
        st.set_page_config("MNIST Classifier", layout="centered")


    def predict_digit(self, img_array):
        img_tensor = self.image_to_tensor(img_array)

        # call the PyTorch model and run the predict method
        prediction = 19
        confidence = 0
        print(img_tensor)
        
        
        return prediction, confidence
    

    def image_to_tensor(self, img):
        #resize image to MNIST format
        img = cv2.resize(img, (28,28), interpolation=cv2.INTER_AREA)
        img = 255 - img  # invert the colors

        # turn to a tensor
        data = torch.from_numpy(img).unsqueeze(0).float() / 255.0
        # print(data.min().item(), data.max().item())

        data = torch.reshape(data, (1, 28, 28))

        print("TENSOR SHAPE:" , data.shape)
        
        return data


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

            st.text(canvas_result.image_data)


            st.image(grey_img, caption="Processed Input", width=100)

            prediction, confidence = self.predict_digit(grey_img)
            st.write(f"PLACEHOLDER {prediction} AND {confidence}%")

            # add the img, prediction, confidence into a dataset




def main():
    web = WebsiteBuild()
    web.homepage()


if __name__ == "__main__":
    main()