import streamlit as st
import pandas as pd
from model.mnist_model import MNIST_CNN
from utilities import train_model, test_model
import os
from os import listdir

# define image ID as state session variable
# initialize image counter
# if "image_id_counter" not in st.session_state:
#     st.session_state.image_id_counter = 0

# if "visibility" not in st.session_state:
#     st.session_state.visibility = "visible"
#     st.session_state.disabled = False

if "refresh_counter" not in st.session_state:
    st.session_state.refresh_counter = 0


class MNIST_Training():
    def __init__(self):
        st.set_page_config("MNIST Training", layout="centered")

    def page_build(self):
        """
        FLOW
        - display the digit from .pt
        - display the guessed label
        - display the model's accuracy for it
        - have buttons where if the label is wrong correct it
        """
        st.title("MNIST Model Trainer")
        st.write("Train the model!")

        # load the data
        data = self.upload_mnist()

        # define all uncorrected images
        uncorrected = data[data["corrected"] == False].reset_index(drop=True)

        if len(uncorrected) == 0:
            st.success("Either all images have been corrected or there are no images to correct.")
            return

        # display gallery
        cols = st.columns(3)  # 3 images per row
        for idx, row in uncorrected.iterrows():
            with cols[idx % 3]:
                image_path = row["img_path"]
                prediction = row["prediction"]
                confidence = row["confidence"]

                st.image(image_path, caption=f"Guess: {prediction}, Conf: {confidence*100:.2f}%", width=150)

                # correct/wrong buttons
                col1, col2 = st.columns(2)
                if col1.button("Correct", key=f"correct_{image_path}"):
                    data.loc[data["img_path"] == image_path, "corrected"] = True
                    data.loc[data["img_path"] == image_path, "true_label"] = prediction
                    data.to_csv("./data/mnist.csv", index=False)
                    st.session_state.refresh_counter += 1  # triggers rerun


                if col2.button("Wrong", key=f"wrong_{image_path}"):
                    st.session_state[f"show_input_{image_path}"] = True

                # show number input for label correction
                if st.session_state.get(f"show_input_{image_path}", False):
                    corrected_label = st.number_input(
                        f"Correct label for {os.path.basename(image_path)}",
                        min_value=0,
                        max_value=9,
                        step=1,
                        key=f"input_{image_path}"
                    )

        if st.button("Save Corrections", key=f"save_{image_path}"):
            data.loc[data["img_path"] == image_path, "true_label"] = corrected_label
            data.loc[data["img_path"] == image_path, "corrected"] = True
            data.to_csv("./data/mnist.csv", index=False)
            st.session_state.refresh_counter += 1  # triggers rerun
            st.success("Dataset saved with all corrections!")

        st.title("Updated MNIST Dataset")
        st.write("Displayed is a data editor where you can see the full DataFrame object and edit any labels if need be.")
        st.data_editor(data, width="stretch", num_rows="dynamic")


    def upload_mnist(self):
        mnist_df = pd.read_csv("./data/mnist.csv")
        print("MNIST Data Loaded")
        return mnist_df


    def data_refinement(self):
        pass

## MAIN
if __name__ == "__main__":
    train_page = MNIST_Training()
    train_page.page_build()