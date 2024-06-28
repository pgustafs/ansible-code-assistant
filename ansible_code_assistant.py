from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.callbacks import StreamlitCallbackHandler
import streamlit as st
from ansible_doc_new import AnsibleDocLoader
from config import *
import logging
import sys
import time
import requests

# Configuration values
RAW_TXT_DOCS_FOLDER = 'raw_txt_docs'

# Set up logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def split_and_embed(documents):
    """Split documents into chunks and embed them using Hugging Face embeddings."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(folder_path=INDEX_PERSIST_DIRECTORY, index_name=INDEX_NAME)

@st.cache_resource(show_spinner=False)
def check_model_service():
    """Check the availability of the model service."""
    start_time = time.time()
    print("Checking Model Service Availability...")
    
    while True:
        try:
            response = requests.get(f'{model_service}/models')
            if response.status_code == 200:
                print("Model Service Available")
                print(f"{time.time() - start_time} seconds")
                return
        except requests.RequestException:
            pass
        time.sleep(1)

def load_vectorstore():

    # load index
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",  # alternatively use "sentence-transformers/all-MiniLM-l6-v2" for a light and faster experience.
        model_kwargs={'device':'cpu'}, 
        encode_kwargs={'normalize_embeddings': True}
    )
    # load the saved embeddings
    vectorstore = FAISS.load_local(folder_path=INDEX_PERSIST_DIRECTORY,embeddings=embeddings,allow_dangerous_deserialization=True,index_name=INDEX_NAME)

    return vectorstore

if __name__ == "__main__":
    urls = [
        'https://docs.ansible.com/ansible/latest/collections/community/libvirt/virt_module.html',
        'https://docs.ansible.com/ansible/latest/collections/ansible/builtin/uri_module.html'
    ]
    
    ansible_doc_loader = AnsibleDocLoader(raw_txt_docs_folder='raw_txt_docs')
    ansible_docs = ansible_doc_loader.load(urls)
    split_and_embed(ansible_docs)

    # retriever interface using vector store. 
    # Use similarity searching algorithm and return 3 most relevant documents.
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    # Set the model service endpoint
    model_service = os.getenv("MODEL_ENDPOINT", "http://localhost:8000")
    model_service = f"{model_service}/v1" 

    with st.spinner("Checking Model Service Availability..."):
        check_model_service()

    st.title("Code Generation App")

    # Initialize chat messages
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
        
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Set up the language model
    llm = ChatOpenAI(base_url=model_service, temperature=0, api_key="EMPTY", streaming=True)

    # Define the Langchain chain
    prompt_template = """
    Try to use the following pieces of context to answer the question at the end. Please follow the following rules:
    1. Always use the Ansible module's FQDN when generating a playbook.
    2. If you don't know the answer, don't try to make up an answer. Just say "I can't find the final answer but you may want to check the following links".
    {context}
                                            
    Question: {input}
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    chain = (
        {"context": retriever, "input": RunnablePassthrough()}
        | prompt
        | llm
    )

    # Process user input and generate response
    if user_input := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").markdown(user_input)
        
        st_callback = StreamlitCallbackHandler(st.container())
        response = chain.invoke(user_input, {"callbacks": [st_callback]})

        st.chat_message("assistant").markdown(response.content)
        st.session_state.messages.append({"role": "assistant", "content": response.content})
        st.rerun()
