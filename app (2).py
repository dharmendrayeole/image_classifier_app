
import streamlit as st
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import pickle
from PIL import Image
st.set_option("deprecation.showfileUploaderEncoding",False)
st.title("Image Classifier using Machine Learning")
st.text("Upload the Image")

model=pickle.load(open("img_model.p","rb"))

uploaded_file= st.file_uploader("Choose an image...",type="jpg")

if uploaded_file is not None:
  img= Image.open(uploaded_file)
  st.image(img,caption="Uploaded Image")

  if st.button("PREDICT"):
    CATEGORIES=['pretty sunflower','rugby ball leather','ice cream cone']
    st.write("Result...")
    flat_data=[]
    img=np.array(img)
    img_resized=resize(img,(150,150,3))
    flat_data.append(img_resized.flatten())
    flat_data=np.array(flat_data)
    y_out=model.predict(flat_data)
    y_out=CATEGORIES[y_out[0]]
    st.title(f,"PREDICTED OUT: {y_out}")
