## MNIST Classifier Application
This application is a Python project where an MNIST digit recognizer model is programmed to guess what a user types in a canvas object. The website is created/coded in `streamlit`, there are 4 different pages to this application:
1) The `home` page is where the user will write their digits and call the model to predict.
2) The `data labeling` page is where the user will go through the compiled digits written and make sure they are correctly labeled, this is the Human-In-The-Loop (HITL) part of this application.
3) The `model training` page is where after a certain amount of corrected digit labels (starting at 500) the user will be able to retrain the model to improve the overall accuracy of the model.
4) The `dashboard` page is where the data analytics is displayed.   

### Libraries Used
These are the libraries defined with in the `requirements.txt` file.
```txt
os
streamlit
streamlit_drawable_canvas
cv2
datetime
matplotlib
```


## PERSONAL NOTES:
- 