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


        st.header("NOTES:")
        st.write("ACCURACY AT SEPT: Test Set: Average Loss: 0.0398, Accuracy: 9879/10000 (98.79%). At this point no retraining has been done.")
        st.write("I want to make a scatter plot of confident scores per month (or not because of the distribution graph)")
        st.write("I can keep my overall digit confidence score and then I can do I per month to show guessing improvement")

## MAIN
if __name__ == "__main__":
    db = MNIST_Dashboard()
    db.build_dashboard()

    # do st.metric(), chart elements