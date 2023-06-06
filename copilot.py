
"""Imports the required modules for the Streamlit web app."""
import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.llms import AzureOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplate import css, bot_template, user_template


def carrega_credenciais():
    """
    Loads the OpenAI credentials from environment variables.

    Returns:
        A tuple containing the OpenAI API type, version, base URL, API key, and model deployment ID.
    """
    # Load environment variables
    load_dotenv()

    # Define OpenAI credentials
    openai_api_type = os.getenv("OPENAI_API_TYPE")
    openai_api_version = os.getenv("OPENAI_API_VERSION")
    openai_api_base = os.getenv("OPENAI_API_BASE")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    model_deployment_id = os.getenv("MODEL_DEPLOYMENT_ID")

    # Set environment variables
    if openai_api_base is not None:
        os.environ["OPENAI_API_BASE"] = openai_api_base
    if openai_api_key is not None:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    if openai_api_type is not None:
        os.environ["OPENAI_API_TYPE"] = openai_api_type
    if openai_api_version is not None:
        os.environ["OPENAI_API_VERSION"] = str(openai_api_version)

    return dict(openai_api_type=openai_api_type, openai_api_version=openai_api_version, openai_api_base=openai_api_base, openai_api_key=openai_api_key, model_deployment_id=model_deployment_id)

def define_embedder():
    """
    Defines and returns an OpenAIEmbeddings object with the specified parameters.

    Returns:
        OpenAIEmbeddings: An OpenAIEmbeddings object with the specified parameters.
    """
    embeddings = OpenAIEmbeddings(
    client=any,
    deployment='embeddings',
    model='text-embedding-ada-002',
    openai_api_type="azure",
    openai_api_base="https://copilot-openai.openai.azure.com",
    openai_api_version="2023-05-15",
    openai_api_key=os.environ["OPENAI_API_KEY"],
    chunk_size=1
    )
    return embeddings


def processa_documentos(pdfs):
    """
    Extracts text from PDF files and concatenates them into a single string.

    Args:
        pdfs (list): A list of PDF files.

    Returns:
        str: A string containing the concatenated text from all PDF files.
    """
    texto = ""
    for pdf in pdfs:
        pdf_reader = PdfReader(pdf)
        for pages in pdf_reader.pages:
            texto += pages.extract_text()
    return texto


def separa_texto(texto):
    """
    Splits a given text into smaller chunks using a RecursiveCharacterTextSplitter.

    Args:
        texto (str): The text to be split.

    Returns:
        list: A list of smaller text chunks.
    """
    separador_de_texto = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separator="\n")
    trechos = separador_de_texto.split_text(texto)
    return trechos


def carrega_vector_db(trechos):
    """
    Loads the embeddings for the given text chunks and creates a FAISS vector store.

    Args:
        trechos (list): A list of text chunks.

    Returns:
        FAISS: A FAISS vector store containing the embeddings for the given text chunks.
    """
    embeddings = define_embedder()
    metadata = [{"source": str(i)} for i in range(len(trechos))]
    vector_store = FAISS.from_texts(trechos, embeddings, metadatas=metadata)
    vector_store.save_local("faiss_index")
    return vector_store

def pega_resposta(query, docs):
    """
    Uses a Langchain QA chain to find the best answer to a given query.

    Args:
        query (str): The query to be answered.
        store (FAISS): A FAISS vector store containing the embeddings for the text chunks.

    Returns:
        dict: A dictionary containing the answer to the query split into two parts: parte1 and parte2.
    """
    llm = AzureOpenAI(model_kwargs={'engine':'text-davinci-003'}, model='text-davinci-003', client=any)
    chain = load_qa_with_sources_chain(llm=llm, chain_type="stuff")
    response = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
    response_parts = response['output_text'].split("SOURCES:")
    parte1 = response_parts[0]
    parte2 = response_parts[1] if len(response_parts) > 1 else ""
    return dict(response=response, parte1=parte1, parte2=parte2)

def conversa(vectorstore):
    """
    Creates a conversational retrieval chain using the given vector store.

    Args:
        vectorstore (FAISS): A FAISS vector store containing the embeddings for the text chunks.

    Returns:
        ConversationalRetrievalChain: A conversational retrieval chain.
    """
    llm = AzureOpenAI(model_kwargs={'engine':'text-davinci-003'}, model='text-davinci-003', client=any)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def gera_conversa(pergunta):
    """
    Generates a conversation response to a given question using the Langchain conversational retrieval chain.

    Args:
        pergunta (str): The question to be answered.

    Returns:
        None
    """
    response = st.session_state.conversation({'question': pergunta})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    """
    The main function that runs the Streamlit web app. It loads the required modules, sets the page configuration, and processes the user's uploaded documents. It also allows the user to search for information within the processed documents.

    Returns:
        None
    """
    carrega_credenciais()
    st.set_page_config(page_title="Analisador de documentos",
                       page_icon=":books:")
    st.write(css,unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Super Analizador de PDFs :books:")

    col1, col2 = st.columns([2, 4])

    col1.subheader("")
    with st.sidebar:
        st.subheader("📖 carregue seus documentos")
        pdf_docs = st.file_uploader(label="uploads", accept_multiple_files=True, type=["pdf"])
        if st.button("Processar"):
            with st.spinner("Processando..."):
                if pdf_docs:
                    texto = processa_documentos(pdf_docs)
                    trechos = separa_texto(texto)
                    store = carrega_vector_db(trechos)
                    st.session_state.conversation = conversa(store)
                    st.success("Processamento concluído!")
                else:
                    st.error("Nenhum documento processado!")

    col2.subheader("")
    pergunta = st.text_input("")

    if pergunta:
        gera_conversa(pergunta)


if __name__ == '__main__':
    main()
