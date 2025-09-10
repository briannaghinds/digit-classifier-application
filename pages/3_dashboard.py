import streamlit as st

class MNIST_Dashboard():
    def __init__(self):
        st.set_page_config("MNIST Dashboard", layout="centered")

    def build_dashboard(self):
        st.title("MNIST Analyst Dashboard")
        st.write("This projects purpose is to visualize the growth of a AI model. To begin the model was already trained on a widely used handwritten digit dataset. The gathering of digits will aid in making the model more accuracte in indentifying messy digit writing.")

        # the graphs will need to be built with R so I can add the images here
        st.header("Non-Technical Graphs")

        st.header("Technical Graphs")


## MAIN
if __name__ == "__main__":
    db = MNIST_Dashboard()
    db.build_dashboard()

    # do st.metric(), chart elements