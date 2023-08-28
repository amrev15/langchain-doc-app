import streamlit as st
import base64
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
import pandas as pd
import requests

def generate_response(uploaded_file, openai_api_key, query_text):
    # Load document if file is uploaded
    if uploaded_file is not None:
        documents = [uploaded_file.read().decode()]
        # Split documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.create_documents(documents)
        # Select embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        # Create a vectorstore from documents
        db = Chroma.from_documents(texts, embeddings)
        # Create retriever interface
        retriever = db.as_retriever()
        # Create QA chain
        qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key), chain_type='stuff', retriever=retriever)
        return qa.run(query_text)

def set_background_image(image_url):
    response = requests.get(image_url)
    img_data = response.content
    img_base64 = base64.b64encode(img_data).decode()
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % img_base64
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Set background image from GitHub repository
image_url = 'https://raw.githubusercontent.com/amrev15/langchain-doc-app/main/environment-clipart-eco-friendly-9.png'
set_background_image(image_url)

# Page title
st.set_page_config(page_title='🦜🔗 Ask the Doc App')
st.title('🦜🔗 Ask the Doc App')

# Check if OpenAI API key is present in secrets management
if 'OPENAI_API_KEY' not in st.secrets:
    st.error('OpenAI API key not found in secrets management. Please add it to use this app.')
else:
    openai_api_key = st.secrets['OPENAI_API_KEY']

# File upload
uploaded_file = st.file_uploader('Upload an article', type=['txt', 'pdf', 'docx'])
# Query text
query_text = st.text_input('Enter your question:', placeholder='Please provide a short summary.', disabled=not uploaded_file)

# Form input and query
result = []
with st.form('myform', clear_on_submit=True):
    submitted = st.form_submit_button('Submit', disabled=not(uploaded_file and query_text))
    if submitted and openai_api_key.startswith('sk-'):
        with st.spinner('Calculating...'):
            response = generate_response(uploaded_file, openai_api_key, query_text)
            result.append(response)
            del openai_api_key

if len(result):
    st.info(response)

# Bot analytics bar chart
topics = ['Topic 1', 'Topic 2', 'Topic 3']
values = [3, 2, 1]
df = pd.DataFrame({'Topic': topics, 'Count': values})
st.bar_chart(df)
