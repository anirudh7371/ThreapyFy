import streamlit as st
import speech_recognition as sr
from textblob import TextBlob
import cv2
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import numpy as np
from pyrebase import initialize_app
import requests
from streamlit_lottie import st_lottie
from email import send_email
from collections import Counter
import pandas as pd
import os


st.set_page_config(page_title="ThearpyFy", page_icon="img.png")

emotion_dict = {0:'angry', 1 :'happy', 2: 'neutral', 3:'sad', 4: 'surprise'}
json_file = open('emotion_model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)
classifier.load_weights("emotion_model1.h5")

api_key=st.secrets["API_KEY"]
load_dotenv()
firebaseConfig = {
  apiKey: os.environ.get("API_KEY"),
  authDomain: "thearpyfy.firebaseapp.com",
  projectId: "thearpyfy",
  storageBucket: "thearpyfy.appspot.com",
  messagingSenderId: "1072598283916",
  appId: "1:1072598283916:web:bc75eb07b0efb8f1f4ada9",
  measurementId: "G-Z0FK0Z1ZV0"
};

firebase = initialize_app(firebaseConfig)
auth = firebase.auth()
db = firebase.database()
storage = firebase.storage()


@st.cache_data
def convert_df(df):
    return df.to_csv().encode('utf-8')

def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


col1,col2= st.columns([1,2])
with col1:
    lottie_animation_1 = "https://assets9.lottiefiles.com/packages/lf20_fCKkXdwO3V.json"
    lottie_anime_json = load_lottie_url(lottie_animation_1)
    st_lottie(lottie_anime_json, key="mental")

with col2:
    st.header("🧠ThearpyFy")

emotion_list=[]
try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            image=img_gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(
                x + w, y + h), color=(255, 0, 0), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_dict[maxindex]
                output = str(finalout)
                emotion_list.append(output)
            label_position = (x, y)
            cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        count = Counter(emotion_list)
        max_count = count.most_common()[0][1]
        return img




st.sidebar.markdown(
    "<h1 style='text-align: center; color: #FFFFFF; font-weight: bold;'>👋 WELCOME ! </h1>",unsafe_allow_html=True)

st.sidebar.markdown("<h1 style='font-size:20px; padding-bottom:0px;'> Login / SignUp to MindDiary</h1>", unsafe_allow_html=True)
choice = st.sidebar.radio('', ['Login', 'SignUp'])
st.write('<style>div.row-widget.stRadio > div {flex-direction:row;}</style>', unsafe_allow_html=True)
st.sidebar.markdown("<h1 style='font-size:20px; padding-bottom:0px;'> Enter your email address:</h1>", unsafe_allow_html=True)
email = st.sidebar.text_input("")
st.sidebar.markdown("<h1 style='font-size:20px; padding-bottom:0px;'> Enter your password:</h1>", unsafe_allow_html=True)
password = st.sidebar.text_input("", type="password")
if choice == "SignUp":
    st.sidebar.markdown("<h1 style='font-size:20px; padding-bottom:0px;'> Please enter your Doctor's Email:</h1>",
                        unsafe_allow_html=True)
    handle = st.sidebar.text_input("", value="doctor@gmail.com")
    submit = st.sidebar.button('Create my Account')
    if submit:
        try:
            user = auth.create_user_with_email_and_password(email, password)
            st.success("Your account is created successfully!")
            st.success("Click on Login to continue")
            st.balloons()
            user = auth.sign_in_with_email_and_password(email, password)
            db.child(user['localId']).child("Handle").set(handle)
            db.child(user['localId']).child("Id").set(user['localId'])

        except Exception as e:
            st.info(e)
            st.info("This account already exists !")
if choice == "Login":
    login = st.sidebar.checkbox('Login', key=2)
    if login:
        try:
            user = auth.sign_in_with_email_and_password(email, password)
            st.subheader("Your Reports:")
            df = pd.DataFrame(
                [
                    {"Mental State": "Neutral", "Date": "03/06/2023"},
                    {"Mental State": "Happy", "Date": "04/06/2023"},
                    {"Mental State": "Neutral", "Date": "05/06/2023"}
                ]
            )
            edited_df = st.data_editor(df)



            doc_email=db.child(user['localId']).child("Handle").get().val()
            st.sidebar.success("You have Logged in Successfully!")
            st.subheader("Upload your audio file:")
            audio_file = st.file_uploader("", type=["wav", "mp3"])
            sentiment=""
            if audio_file is not None:
                recognizer = sr.Recognizer()
                audio = sr.AudioFile(audio_file)
                with audio as source:
                    audio_data = recognizer.record(source)
                    text = recognizer.recognize_google(audio_data)

                blob = TextBlob(text)
                sentiment_score = blob.sentiment.polarity

                st.markdown("<h1 style='font-size:20px;'>Sentiment Analysis Result:</h1>", unsafe_allow_html=True)
                st.markdown(f"<h1 style='font-size: 15px; font-weight: bold;'>Text:</h1>{text}", unsafe_allow_html=True)
                if sentiment_score >= -0.5 and sentiment_score <= 0.5:
                    sentiment="Neutral"
                elif sentiment_score > 0.5:
                    sentiment = "Positive"
                else:
                    sentiment = "Negative"
                st.write(f"<h1 style='font-size: 15px;font-weight:bold;'>Sentiment: {sentiment}</h1> ", unsafe_allow_html=True)


            st.subheader("Webcam Live Sentiment Analysis")
            st.markdown("<h5>Click on start to use webcam and detect your face emotion</h5>",unsafe_allow_html=True)
            lst=webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
            if lst and sentiment:
                report=sentiment
                send=st.button("Send report to doctor")
                if send:
                    auth=st.secrets["AUTH_TOKEN"]
                    try:
                        succ=send_email(doc_email,report,auth)
                    except:
                        st.info("Oops! Something went wrong. Please try again later.")
                    if succ:
                        st.success("Report sent successfully!")
        except Exception as e:
            st.write(e)
            st.info("Incorrect Email/Password or Account doesn't exist!")




col1,col2,col3= st.sidebar.columns([1,5,1])
with col2:
    lottie_animation_1 = "https://assets10.lottiefiles.com/packages/lf20_uui8d4hv.json"
    lottie_anime_json = load_lottie_url(lottie_animation_1)
    st_lottie(lottie_anime_json, key="hello")


def write_bytesio_to_file(filename, bytesio):
    with open(filename, "wb") as outfile:
        outfile.write(bytesio.getbuffer())
