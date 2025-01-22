import streamlit as st
import requests
from PIL import Image
# to run quickly use "streamlit run frontend.py"

st.title(':robot_face: Pneumonia Detection Model using timm')
st.info('This app can be used to detect pneumonia from chest X-ray images. Just drop the X-Ray image below!')

# File uploader
img_data = st.file_uploader(label='Upload an X-Ray image', type=['png', 'jpg', 'jpeg'])

prediction = None

if img_data is not None:
    # Display the uploaded image
    uploaded_img = Image.open(img_data)
    st.image(uploaded_img, caption='Provided Image')

    # Send the image to FastAPI endpoint
    # ----------------------------------
    # IMPORTANT: replace http://localhost:8000 with your actual FastAPI URL and port
    with st.spinner('Predicting...'):
        try:
            files = {"file": (img_data.name, img_data, img_data.type)}
            response = requests.post("http://localhost:8000/predict/", files=files)

            if response.status_code == 200:
                result = response.json()
                prediction = result["predicted_class"]
                probabilities = result["probabilities"]
            else:
                st.error(f"Prediction failed: {response.text}")
        except Exception as e:
            st.error(f"Error occurred: {e}")

# Display the prediction result
if prediction is None:
    st.write('Please upload an image to get the prediction!')
else:
    st.write('Our (very smart) model carefully examined the X-Ray. The likely diagnosis is:')
    st.write(prediction)

    if prediction == 0:
        st.success('Lucky you, everything seems fine!')
    elif prediction == 1:
        st.warning('Our team is wishing you a speedy recovery!')
