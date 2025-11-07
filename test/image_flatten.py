import os
import numpy as np
import pandas as pd
from PIL import Image

# Folder containing images
img_folder = "images"
img_files = sorted(os.listdir(img_folder))  # Make sure sorted order matches your labels

# Optional: your labels in the same order
labels = pd.read_csv("./data/mnist.csv")["true_label"]

data = []

for file in img_files:
    try:
        img_path = os.path.join(img_folder, file)
        img = Image.open(img_path).convert("L")  # convert to grayscale
        img = img.resize((28, 28))
        data.append(np.array(img).flatten())
    except Exception as e:
        print(f"Failed: {file}, {e}")

# Convert to DataFrame
df = pd.DataFrame(data)
df["label"] = labels  # add labels as a column

# Save to CSV
df.to_csv("mnist_images_flat.csv", index=False)
