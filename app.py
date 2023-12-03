# Importing necessary libraries and modules
import streamlit as st  # Streamlit for creating web apps
from streamlit_chat import message  # Streamlit chat for creating chat interfaces
from langchain.embeddings.openai import OpenAIEmbeddings  # OpenAI embeddings for NLP tasks
from langchain.chat_models import ChatOpenAI  # Chat model from Langchain for OpenAI-based chat
from langchain.chains import ConversationalRetrievalChain  # A chain for conversational retrieval
from langchain.document_loaders.csv_loader import CSVLoader  # Loader for CSV documents
from langchain.vectorstores import FAISS  # FAISS for efficient similarity search
import tempfile  # For creating temporary files
import os  # For interacting with the operating system

# Secure file handling from bunkoer library
from bunkoer.security import SecureFile


# Sidebar uploader for CSV files
uploaded_file = st.sidebar.file_uploader("upload", type="csv")

# Handling the uploaded file
if uploaded_file:
    _, file_extension = os.path.splitext(uploaded_file.name)  # Extracting file extension
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())  # Writing the uploaded file to a temporary file
        tmp_file_path = tmp_file.name  # Storing the path of the temporary file
        file_secure = SecureFile(tmp_file_path)  # Securing the temporary file
        try:
            loader = CSVLoader(file_path=file_secure, encoding="utf-8")  # Trying to load the file with utf-8 encoding
            data = loader.load()
        except:
            loader = CSVLoader(file_path=file_secure, encoding="cp1252")  # Fallback to cp1252 encoding if utf-8 fails
            data = loader.load()

    loader = CSVLoader(file_path=file_secure, encoding="utf-8", csv_args={'delimiter': ','})  # Loading the CSV file
    data = loader.load()

    embeddings = OpenAIEmbeddings()  # Initializing OpenAI embeddings
    vectorstore = FAISS.from_documents(data, embeddings)  # Creating a vector store from the data and embeddings

    # Setting up the conversational retrieval chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0.7, model_name='gpt-4-1106-preview'),  # Configuring the chat model
        retriever=vectorstore.as_retriever()  # Setting up the retriever with the vector store
    )

    # Initializing session state for secure file
    if 'secure_file' not in st.session_state:
        st.session_state['generated'] = ["Hello ! Your file is now secure to use on any AI, Ask me anything about it " + uploaded_file.name + " 🤗"]

# Function for handling conversational chat
def conversational_chat(query):
    result = chain({"question": query, "chat_history": st.session_state['history']})  # Passing the query and history to the chain
    st.session_state['history'].append((query, result["answer"]))  # Updating the chat history
    return result["answer"]  # Returning the answer

# Initializing various states for the conversation
if 'history' not in st.session_state:
    st.session_state['history'] = []
if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello ! Upload a file for begin "]
if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey ! 👋"]

# Setting up the response container in the Streamlit app
response_container = st.container()
container = st.container()

# Creating the chat interface
with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("Query:", placeholder="Talk about your csv data here (:",
                                   key='input')  # Input field for user query
        submit_button = st.form_submit_button(label='Send')  # Send button
        if submit_button and user_input:
            output = conversational_chat(user_input)  # Handling the chat functionality
            st.session_state['past'].append(user_input)  # Updating past queries
            st.session_state['generated'].append(output)  # Updating generated responses

# Displaying the conversation history
if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user',
                     avatar_style="big-smile")  # Displaying user's messages
            message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")  # Displaying AI's responses

