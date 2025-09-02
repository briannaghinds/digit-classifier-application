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

if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

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

        # --- SESSION STATE ---
        if "current_idx" not in st.session_state:
            st.session_state.current_idx = 0
        if "show_correction" not in st.session_state:
            st.session_state.show_correction = False

        # iterate through image folder
        folder_dir = "./images"
        for image in os.listdir(folder_dir):
            if not image.endswith(".png"):
                continue

            image_path = f"./images/{image}"

            # pull the rest of the data based on image path value
            row = data.loc[data["img_path"] == image_path]
            if row.empty:
                continue  # skip if not in dataset

            prediction = row["prediction"].values[0]
            confidence = row["confidence"].values[0]
            corrected = row["corrected"].values[0]
            print(corrected)

            # show image
            st.image(image_path, caption=f"Model Guess: {prediction}, Confidence: {confidence*100:.2f}")

            if not corrected:
                st.write("Looking at the image's caption, was the model wrong or correct in this instance?")
                col1, col2 = st.columns(2)

                if col1.button(f"{image} is Correct", key=f"btn_correct_{image}"):
                    st.success("Marked as correct")
                    data.loc[data["img_path"] == image_path, "corrected"] = True

                if col2.button(f"{image} is Wrong", key=f"btn_wrong_{image}"):
                    corrected_label = st.number_input(
                        f"Correct label for {image}?",
                        min_value=0,
                        max_value=9,
                        step=1,
                        key=f"input_{image}"
                    )
                    if st.button(f"Save {image}", key=f"save_{image}"):
                        st.success(f"Saved correction → {corrected_label}")
                        data.loc[data["img_path"] == image_path, "prediction"] = corrected_label
                        data.loc[data["img_path"] == image_path, "corrected"] = True

            else:
                st.info(f"Already corrected: {corrected}")

        # save the new updated data
        if st.button("Save Dataset"):
            data.to_csv("./mnist_corrected.csv", index=False)
            st.success("Dataset saved with all corrections!")


        # # st.image("", caption="")
        # # st.write("IMAGE HERE")  # WILL DELETE LATER

        # # pull the rest of the data given the image path
        # data = self.upload_mnist()
        # row = data.loc[data["img_path"] == f"./images/{image}"]

        # col1, col2 = st.columns(2)
        # col1.write(f"Model's Guess: {row['prediction'][0]}")
        # col2.write(f"Model's Confidence {row['confidence'][0]*100:.2f}%")


        # correct = col1_2.button("Correct", key="disabled")
        # wrong = col2_2.button("Wrong", key="visibility")

        # if correct:
        #     print("YAY THE MODEL GUESSED CORERECT")

        # # if wrong:
        # # st.write("What is the correct label?")
        # corrected_label = st.number_input(
        #     "What is the correct label?", 
        #     0, 
        #     9, 
        #     label_visibility="visible",
        #     disabled=st.session_state.disabled
        # )
        # st.write(corrected_label)
        # print(f"CORRECTED LABEL: {corrected_label}")
        # st.write(f"Image Label has been changed to: {corrected_label}")



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