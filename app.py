import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px
import platform
plt=platform.system()
if plt == 'Windows': pathlib.PosixPath = pathlib.WindowsPath
st.title('Transport classification model')
file=st.file_uploader('Upload Picture', type=['png','jpeg', 'svg', 'gif'])
if file:
    st.image(file)
    model=load_learner('transport_model.pkl')
    img=PILImage.create(file)
    pred,pr_id,prob=model.predict(img)
    st.success(f'Prediction: {pred}')
    st.text(f'Probability: {prob[pr_id]*100:.1f}%')
    
    fig=px.bar(x=prob, y=model.dls.vocab)
    st.plotly_chart(fig)
