from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
load_dotenv()

chat_model = ChatOpenAI()

import streamlit as st

st.title('인공지능 시인')

content = st.text_input('시의 주제를 말씀해주세요', '코딩')

if st.button('시 작성 요청하기'):
    with st.spinner("시 작성 중"):
        result = chat_model.invoke(content + "에 대한 시를 써줄래?")
        st.write(result.content)
