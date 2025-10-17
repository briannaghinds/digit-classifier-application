# this will be a simple script that will go through the mnist dataset and make sure the digit and true label are correct

import pandas as pd
import cv2
import matplotlib.pyplot as plt

# import the data
df = pd.read_csv("./data/mnist.csv")  # finished first 600 so from forward will need to double check 600+

# going through the dataframe
for i in range(len(df)):
    image_path = df["img_path"][i]
    print(image_path)
    true_label = df["true_label"][i]
    print(true_label)

    img_array = cv2.imread(image_path)

    plt.imshow(img_array)
    plt.title(f"{image_path} TRUE LABEL DEFINED AS: {true_label}")
    plt.show()