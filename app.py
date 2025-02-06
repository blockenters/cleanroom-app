import streamlit as st
import altair as alt
import pandas as pd
from datetime import datetime
import json
import os

from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np


def load_history():
    if os.path.exists('analysis_history.json'):
        with open('analysis_history.json', 'r') as f:
            return json.load(f)
    return []

def save_history(result_text, confidence):
    history = load_history()
    history.append({
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'result': result_text,
        'confidence': confidence
    })
    with open('analysis_history.json', 'w') as f:
        json.dump(history, f)

def main():
    # 페이지 기본 설정
    st.set_page_config(
        page_title="방 청결도 분석기",
        page_icon="🏠",
        layout="wide"
    )
    
    # 사이드바 대시보드
    with st.sidebar:
        st.title('🏠 방 청결도 분석기')
        st.markdown("---")
        
        st.markdown("### 👀 분석 가능한 방 예시")
        
        # 깨끗한 방 예시
        st.image("images/clean.png", caption="깨끗한 방", width=200)
        st.success("✨ 깔끔하게 정리된 방")
        st.markdown("""
        - 물건들이 제자리에 있음
        - 침대가 정리되어 있음
        - 전반적으로 깔끔한 상태
        """)
        
        st.markdown("---")
        
        # 지저분한 방 예시
        st.image("images/messy.png", caption="지저분한 방", width=200)
        st.warning("⚠️ 정리가 필요한 방")
        st.markdown("""
        - 물건들이 흩어져 있음
        - 침대가 정리되지 않음
        - 전반적으로 어수선한 상태
        """)
        
        st.markdown("---")
        st.caption("© 2025 방 청결도 분석기")
        st.caption("AI 기반 이미지 분석 서비스")
        st.caption("Blockenters")
    
    # 메인 화면 전체를 컨테이너로 감싸기
    main_container = st.container()
    with main_container:
        col1, main_col, col3 = st.columns([1,2,1])  # 3등분해서 가운데 column만 사용
        
        with main_col:
            st.title('방 청결도 분석')
            st.markdown("""
            ### AI가 당신의 방이 얼마나 깨끗한지 분석해드립니다!
            방의 전체적인 모습이 잘 보이는 사진을 업로드해주세요.
            """)
            
            # 이미지 업로드 섹션
            st.subheader('📸 방 사진 업로드')
            image = st.file_uploader(
                '깨끗한 사진일수록 정확한 분석이 가능합니다.',
                type=['jpg','png','jpeg']
            )

            if image is not None:
                # 이미지와 분석 결과
                img_col, result_col = st.columns(2)
                
                with img_col:
                    st.markdown("### 업로드된 이미지")
                    st.image(image, use_container_width=True)

                with result_col:
                    st.markdown("### 분석 결과")
                    with st.spinner('AI가 열심히 분석중입니다...'):
                        # 이미지 처리 및 예측
                        image_pil = Image.open(image)
                        model = load_model("model/keras_model.h5", compile=False)
                        class_names = open("model/labels.txt", "r", encoding='utf-8').readlines()

                        # 이미지 전처리
                        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
                        size = (224, 224)
                        image_processed = ImageOps.fit(image_pil, size, Image.Resampling.LANCZOS)
                        image_array = np.asarray(image_processed)
                        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
                        data[0] = normalized_image_array

                        # 예측
                        prediction = model.predict(data)
                        index = np.argmax(prediction)
                        class_name = class_names[index]
                        confidence_score = prediction[0][index]

                        # 결과 표시
                        result_text = class_name[2:].strip()
                        confidence_percentage = float(confidence_score) * 100

                        if "깨끗한" in result_text:
                            st.success(f"🌟 분석 결과: {result_text}")
                        else:
                            st.warning(f"⚠️ 분석 결과: {result_text}")

                        st.progress(confidence_percentage / 100)
                        st.caption(f"신뢰도: {confidence_percentage:.1f}%")

                        # 결과 저장
                        save_history(result_text, confidence_percentage)

                # 통계 섹션
                st.markdown("---")
                st.subheader("📊 분석 통계")
                
                # 데이터 로드 및 DataFrame 변환
                history = load_history()
                if history:
                    df = pd.DataFrame(history)
                    df['date'] = pd.to_datetime(df['timestamp']).dt.date
                    
                    # 통계 차트들을 3개의 컬럼으로 배치
                    stat_col1, stat_col2, stat_col3 = st.columns(3)
                    
                    with stat_col1:
                        st.subheader("방 상태 분포")
                        room_counts = df['result'].value_counts()
                        fig_pie = alt.Chart(pd.DataFrame({
                            'category': room_counts.index,
                            'count': room_counts.values
                        })).mark_arc().encode(
                            theta='count',
                            color='category',
                            tooltip=['category', 'count']
                        ).properties(width=200, height=200)
                        st.altair_chart(fig_pie)
                    
                    with stat_col2:
                        st.subheader("신뢰도 트렌드")
                        recent_df = df.tail(10)
                        confidence_trend = alt.Chart(recent_df).mark_line().encode(
                            x='timestamp',
                            y=alt.Y('confidence', title='신뢰도 (%)')
                        ).properties(width=200, height=200)
                        st.altair_chart(confidence_trend)
                    
                    with stat_col3:
                        st.subheader("시간대별 분석")
                        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
                        hourly_counts = df['hour'].value_counts().sort_index()
                        hourly_chart = alt.Chart(pd.DataFrame({
                            'hour': hourly_counts.index,
                            'count': hourly_counts.values
                        })).mark_bar().encode(
                            x=alt.X('hour:O', title='시간'),
                            y=alt.Y('count:Q', title='분석 횟수')
                        ).properties(width=200, height=200)
                        st.altair_chart(hourly_chart)

            # 하단 설명 섹션
            st.markdown("---")
            st.markdown("""
            ### 💡 사용 팁
            1. **밝은 조명**: 방 전체가 잘 보이도록 밝은 조명에서 촬영하세요
            2. **전체 구도**: 방의 전체적인 모습이 잘 보이게 찍어주세요
            3. **선명도**: 흔들리지 않고 선명한 사진이 좋습니다
            """)

if __name__ == '__main__':
    main()



