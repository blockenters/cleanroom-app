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
        layout="wide"  # 레이아웃을 wide로 변경
    )
    
    # 헤더 섹션
    st.title('🏠 방 청결도 분석기')
    st.markdown("""
    ### AI가 당신의 방이 얼마나 깨끗한지 분석해드립니다!
    """)
    
    # 예시 이미지로 설명하는 섹션
    st.markdown("### 👀 이런 사진을 분석할 수 있어요!")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image("images/clean_room.jpg", caption="깨끗한 방의 예시", use_column_width=True)
        st.success("✨ 깔끔하게 정리된 방")
        st.markdown("""
        - 물건들이 제자리에 있음
        - 침대가 정리되어 있음
        - 전반적으로 깔끔한 상태
        """)
    
    with col2:
        st.image("images/messy_room.jpg", caption="지저분한 방의 예시", use_column_width=True)
        st.warning("⚠️ 정리가 필요한 방")
        st.markdown("""
        - 물건들이 흩어져 있음
        - 침대가 정리되지 않음
        - 전반적으로 어수선한 상태
        """)
    
    # 구분선 추가
    st.markdown("---")
    
    # 사용 방법 설명
    st.markdown("""
    ### 🎯 사용 방법
    1. 방 전체가 잘 보이는 사진을 준비하세요
    2. 아래 업로드 버튼을 통해 사진을 올려주세요
    3. AI가 자동으로 방의 청결 상태를 분석해드립니다
    """)
    
    # 이미지 업로드 섹션
    st.subheader('📸 방 사진 업로드')
    image = st.file_uploader(
        '깨끗한 사진일수록 정확한 분석이 가능합니다.',
        type=['jpg','png','jpeg']
    )

    if image is not None:
        # 이미지 표시 컬럼
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 업로드된 이미지")
            st.image(image, use_column_width=True)

        with col2:
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

                # 통계 섹션 추가
                st.markdown("---")
                st.subheader("📊 분석 통계")
                
                # 데이터 로드
                history = load_history()
                if history:
                    df = pd.DataFrame(history)
                    df['date'] = pd.to_datetime(df['timestamp']).dt.date
                    
                    # 1. 깨끗한/더러운 방 비율 파이 차트
                    col1, col2 = st.columns(2)
                    with col1:
                        room_counts = df['result'].value_counts()
                        st.subheader("방 상태 분포")
                        fig_pie = alt.Chart(pd.DataFrame({
                            'category': room_counts.index,
                            'count': room_counts.values
                        })).mark_arc().encode(
                            theta='count',
                            color='category',
                            tooltip=['category', 'count']
                        ).properties(width=200, height=200)
                        st.altair_chart(fig_pie)
                    
                    # 2. 최근 분석 신뢰도 트렌드
                    with col2:
                        st.subheader("신뢰도 트렌드")
                        recent_df = df.tail(10)  # 최근 10개 결과
                        confidence_trend = alt.Chart(recent_df).mark_line().encode(
                            x='timestamp',
                            y=alt.Y('confidence', title='신뢰도 (%)')
                        ).properties(width=300, height=200)
                        st.altair_chart(confidence_trend)
                    
                    # 3. 시간대별 분석 횟수
                    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
                    hourly_counts = df['hour'].value_counts().sort_index()
                    
                    st.subheader("시간대별 분석 횟수")
                    hourly_chart = alt.Chart(pd.DataFrame({
                        'hour': hourly_counts.index,
                        'count': hourly_counts.values
                    })).mark_bar().encode(
                        x=alt.X('hour:O', title='시간'),
                        y=alt.Y('count:Q', title='분석 횟수'),
                        tooltip=['hour', 'count']
                    ).properties(width=600, height=200)
                    st.altair_chart(hourly_chart)

    # 하단 설명 섹션
    st.markdown("---")
    st.markdown("""
    ### 💡 사용 팁
    1. **밝은 조명**: 방 전체가 잘 보이도록 밝은 조명에서 촬영하세요
    2. **전체 구도**: 방의 전체적인 모습이 잘 보이게 찍어주세요
    3. **선명도**: 흔들리지 않고 선명한 사진이 좋습니다
    """)

    # 푸터
    st.markdown("---")
    st.caption("© 2025 방 청결도 분석기 | AI 기반 이미지 분석 서비스 | Blockenters")

if __name__ == '__main__':
    main()



