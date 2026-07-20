import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
from cv2 import CascadeClassifier
import streamlit as st
from PIL import Image
import pandas as pd
import altair as alt
st.set_page_config(page_title="Face Emotion Recognition", page_icon="🕵️‍♂️")
st.title(" Face Emotion Recognition App")
st.write("Captures image through webcam and infer emotion on it.")

paths = {'model': 'model/emotion_cnn.keras',
        'cascade': 'model/haarcascade_frontalface_alt.xml'}

class_names = ["💢angry", "😭sad", "😲surprise", "😃happy", "😨fear"]

@st.cache_resource
def load_artifacts():
    try:
        model = load_model(paths["model"])
        cascade = CascadeClassifier( paths["cascade"])
        return model , cascade
    except Exception as e:
        print("error occured while loading: {e}")

cnn_model , cascade = load_artifacts()

def _preprocess_face(face_img:np.ndarray):
    face_img = cv2.cvtColor(face_img,cv2.COLOR_RGB2GRAY)
    face_img = cv2.resize(face_img,(48,48))
    face_img = face_img.astype('float32')/255.0
    face_img = face_img.reshape((1,48,48,1))
    return face_img

def _detect_face(frame:np.ndarray, cascade:CascadeClassifier):
    face_img = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    face = cascade.detectMultiScale(face_img, scaleFactor =1.1, minNeighbors=3)
    if len(face) == 0:
        return None
    x,y,w,h = max(face, key= lambda f: f[2]*f[3])
    return x,y,w,h

captured_img = st.camera_input("click to take a picture!!")
if captured_img is not None:
    image = Image.open(captured_img)
    image = np.array(image)
    face_cord = _detect_face( image, cascade)
    if face_cord is None:
        st.warning("couldn't find a face ")
    else:
        x,y,w,h = face_cord
        processed_img = _preprocess_face(image[y:y+h,x:x+w])
        pred = cnn_model.predict(processed_img,verbose=0)
        pred_idx = np.argmax(pred[0])
        confidence = pred[0][int(pred_idx)] * 100
        emotion = class_names[int(pred_idx)]
        st.success(f"Predicted Emotion: {emotion}  (with confidence:{confidence:.1f}%)")

        rec_over_img = cv2.rectangle(image,(x,y),(x+w,y+h) , (0,255,0),3)
        cv2.putText(img=rec_over_img,text=f"{emotion[1:]}", org=(x+100,y-10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6 ,color= (255, 255, 0), thickness=2)
        emotion_df = pd.DataFrame({
            "Emotion": [e.capitalize() for e in class_names],
            "Confidence (%)": [float(p * 100) for p in pred[0]]
        })

        with st.sidebar:
            st.write('### Confidence scores per emotion:')
            chart = alt.Chart(emotion_df).mark_bar(cornerRadiusEnd=4).encode(
                x=alt.X('Confidence (%):Q', scale=alt.Scale(domain=[0, 100])),
                y=alt.Y('Emotion:N', sort='-x', title=''), 
                color=alt.Color('Emotion:N', legend=None),
                tooltip=[alt.Tooltip('Emotion:N'), alt.Tooltip('Confidence (%):Q', format='.1f')]
            ).properties(height=300)
            st.altair_chart(chart, use_container_width=True)
            st.write('### view of captured face:')
            st.image(rec_over_img, caption="face detected", width='stretch')
    

        
if __name__ == "__main__":
    pass
