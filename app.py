import streamlit as st

from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np


def main():
    st.title('깨끗한 방인지 더러운 방인지!')
    st.info('방 사진을 업로드하면, 깨끗한 방인지 더러운 방인지 알려드립니다.')

    file = st.file_uploader('이미지 파일을 업로드하세요.', type=['jpg','png','jpeg']) 

    if file is not None:

        st.image(file)




if __name__ == '__main__':
    main()



