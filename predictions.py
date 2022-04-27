import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from keras.preprocessing import image
from PIL import Image
import streamlit as st
st.set_page_config(page_title='African Wildlife', page_icon='favicon.png')

#Function to load uploaded image
def load_image(image_file):
	img = Image.open(image_file)
	return img
# Function to check the image
def set_bg_hack_url():
    '''
    A function to unpack an image from url and set as bg.
    Returns
    -------
    The background.
    '''
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url("https://wallpaperaccess.com/full/2757600.jpg");
             background-size: cover;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

@st.cache(ttl=48*3600)
def check():

    lr = keras.models.load_model('weights.h5')
    #Prediction Pipeline
    class Preprocessor(BaseEstimator, TransformerMixin):
        def fit(self,img_object):
            return self
        
        def transform(self,img_object):
            img_array = image.img_to_array(img_object)
            expanded = (np.expand_dims(img_array,axis=0))
            return expanded

    class Predictor(BaseEstimator, TransformerMixin):
        def fit(self,img_array):
            return self
        
        def predict(self,img_array):
            probabilities = lr.predict(img_array)
            predicted_class = ['Buffalo', 'Elephant', 'Rhino', 'Zebra'][probabilities.argmax()]
            return predicted_class

    full_pipeline = Pipeline([('preprocessor',Preprocessor()),
                            ('predictor',Predictor())])
    return full_pipeline

def output(full_pipeline,img):
   a=  img
   #a = img.decode('utf-8', 'ignore') 
   a= a.resize((256,256))
   predic = full_pipeline.predict(a)
   return(predic)

def home():
  st.write("## CNN Image Detection Using EfficientNetB6 Model")
  st.write("#### We passed this image for prediction through our EfficientNetB6 model")
  st.image("original.jpg", width=800)
  st.write("#### This is the output, our model predicted")
  st.image("predict111.jpg", width=800)
#   st.image("predictions_img2.jpg", width=800)
  st.write('## To know more about the project, visit the About page')

def about():
  st.write("# Team SMOTE with Real-Time Animal Detection using CNN")
  st.write("### All the files for this project can be found on our Github Repository [here](https://github.com/heyakshayhere/Hamoye_capstone_project_smote/)")


def predict():
    st.subheader('Upload either Buffalo/Elephant/Rhino/Zebra image for prediction')
    image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
    # code for Prediction
    prediction = ''

    # creating a button for Prediction

    if st.button('Predict'):
        if image_file is not None:
            # To See details
            with st.spinner('Loading Image and Model...'):
                full_pipeline = check()
            file_details = {"filename":image_file.name, "filetype":image_file.type,"filesize":image_file.size}
            st.write(file_details)
            img = load_image(image_file)
            st.image(img,width=256)
            with st.spinner('Predicting...'):
                prediction = output(full_pipeline,img)
            st.success(prediction)

def main():
    hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden; }
        footer {visibility: hidden;}
        </style>
        """
    st.markdown(hide_menu_style, unsafe_allow_html=True)
    # giving a title
   
    st.write('<style>.css-10trblm{ text-align:center;} .css-12oz5g7 { max-width: 50rem;} div.row-widget.stRadio > div{flex-direction:row;justify-content: space-evenly;} div.row-widget.stRadio > div > label > div{font-size:24px !important;}</style>',unsafe_allow_html=True)
    st.title('African Wildlife Animal Classifier')
    page = st.radio("", ["Home","Predict","About"])
    # Website flow logic    
    set_bg_hack_url()
    if page == "Home":
        
        home()
    elif page == "Predict":
        predict()
    elif page == "About":
        set_bg_hack_url()
        about()
   
if __name__ == '__main__':
    main()
