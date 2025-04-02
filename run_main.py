from langchain.chains import LLMChain
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
import boto3
import os
import streamlit as st

from langchain.vectorstores import Chroma
from langchain.embeddings.ollama import OllamaEmbeddings
ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

persist_directory = './docs/sample1/'
embedding = OllamaEmbeddings(model="mxbai-embed-large")
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

from langchain.chat_models import ChatOllama
#llm_name = "deepseek-r1:1.5b"
llm_name = st.sidebar.selectbox("Model", ["deepseek-r1:1.5b", "llama3.2:1b"])
temperature = st.sidebar.slider("Temperature",0.0,1.0,0.5,0.01)
print(llm_name)

llm = ChatOllama(temperature=temperature,model=llm_name)
from langchain.chains import RetrievalQA

def my_chatbot(freeform_text):
    prompt = PromptTemplate(
        input_variables="freeform_text",
        template="You are a chatbot. Answer the question carefully. If you dont know say I dont know.\n\n{freeform_text}"
    )

    qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever()
    )

    response=qa_chain.run(freeform_text)
    return response


st.image("./docs/image/pony.jpeg")
st.title("AskPeruna")

freeform_text = st.sidebar.text_area(label="Ask Peruna?",max_chars=100)

if freeform_text:
    response = my_chatbot(freeform_text)
    st.write(response)
