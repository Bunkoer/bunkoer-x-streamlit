import streamlit as st
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
import tempfile
import os

from bunkoer.security import SecureFile

user_api_key = st.sidebar.text_input(
    label="#### Your OpenAI API key 👇",
    placeholder="Paste your openAI API key, sk-",
    type="password"
)

uploaded_file = st.sidebar.file_uploader("upload", type="csv")

if uploaded_file:
    _, file_extension = os.path.splitext(uploaded_file.name)
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
        file_secure = SecureFile(tmp_file_path)
        try:
            loader = CSVLoader(file_path=file_secure, encoding="utf-8")
            data = loader.load()
        except:
            loader = CSVLoader(file_path=file_secure, encoding="cp1252")
            data = loader.load()

    loader = CSVLoader(file_path=file_secure, encoding="utf-8", csv_args={'delimiter': ','})
    data = loader.load()

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(data, embeddings)

    chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(temperature=0.7, model_name='gpt-4-1106-preview'),  # or 'gpt-4-32k', 'gpt-4-32k-0613', etc.
    retriever=vectorstore.as_retriever()
    )

    if 'secure_file' not in st.session_state:
        st.session_state['generated'] = ["Hello ! Your file is now secure to use on any AI, Ask me anything about it " + uploaded_file.name + " 🤗"]

def conversational_chat(query):
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]

if 'history' not in st.session_state:
    st.session_state['history'] = []
if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello ! Upload a file for begin "]
if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey ! 👋"]

response_container = st.container()
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("Query:", placeholder="Talk about your csv data here (:",
                                   key='input')
        submit_button = st.form_submit_button(label='Send')
        if submit_button and user_input:
            output = conversational_chat(user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user',
                     avatar_style="big-smile")
            message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")

