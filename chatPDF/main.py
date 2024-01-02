__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import tempfile
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import streamlit as st


st.title("ChatPDF")
st.write("---")
uploaded_file = st.file_uploader("Choose a file")

def pdf_to_document(upload_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, upload_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(upload_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)

    # load_dotenv()

    # Loader
    loader = PyPDFLoader("good.pdf")
    pages = loader.load_and_split()


    # Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=300,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    texts = text_splitter.split_documents(pages)

    # Embedding
    from langchain.embeddings import OpenAIEmbeddings
    embeddings_model = OpenAIEmbeddings()

    # Load Chroma
    from langchain.vectorstores import Chroma
    db = Chroma.from_documents(texts, embeddings_model)

    # Question
    st.header("PDF에게 질문해보세요!")
    question = st.text_input("질문을 입력하세요!")
    # question = "김첨지는 무슨 일을 하는 사람이야?"

    if st.button("질문하기"):
        with st.spinner("기다려주세요"):
            llm = ChatOpenAI(temperature=0.5)
            # retriever_from_llm = MultiQueryRetriever.from_llm(
            #     retriever=db.as_retriever(), llm=llm
            # )

            # result = retriever_from_llm.get_relevant_documents(query=question)
            # print(result)


            qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
            st.write(qa_chain({"query": question})["result"])


