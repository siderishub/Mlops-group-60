import streamlit as st
from PIL import Image

img_size = 128

st.title(':robot_face:Pneumonia Detection Model using timm')

st.info('This app can be used to detect pneumonia from chest X-ray images')


img_data = st.file_uploader(label='load the image', type=['png', 'jpg', 'jpeg'])

if img_data is not None:
    
    # display the image 
    
    uploaded_img = Image.open(img_data)
    st.image(uploaded_img, caption='Provided Image')
    
    # load image file for prediction
    
    img_path = f'./{img_data.name}'
    img = img.load_imag(img_path), target_size=(img_size, img_size)
    
    prediction = None # Make model.predict(img)
    
st.title('Our (very smart) model carefully examined the X-Ray and the likely diagnosis is:')
st.write(prediction)

if prediction == 0:
    st.write('Lucky you, everything seems fine!')
    
if prediction ==1:
    st.write('Our team is wishing a speedy recovery!')


