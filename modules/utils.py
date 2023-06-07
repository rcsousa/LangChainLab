import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.llms import AzureOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplate import bot_template, user_template



def carrega_credenciais():
    """
    Carrega as credenciais da OpenAI das variáveis de ambiente.

    Retorna:
        Uma tupla contendo o tipo de API da OpenAI, a versão, a URL base, a chave da API e o ID de implantação do modelo.
    """
    # Carrega as variáveis de ambiente
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
    Define e retorna um objeto OpenAIEmbeddings com os parâmetros especificados.

    Retorna:
        OpenAIEmbeddings: Um objeto OpenAIEmbeddings com os parâmetros especificados.
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
    Extrai o texto de arquivos PDF e os concatena em uma única string.

    Args:
        pdfs (list): Uma lista de arquivos PDF.

    Returns:
        str: Uma string contendo o texto concatenado de todos os arquivos PDF.
    """
    texto = ""
    for pdf in pdfs:
        pdf_reader = PdfReader(pdf)
        for pages in pdf_reader.pages:
            texto += pages.extract_text()
    return texto

def separa_texto(texto):
    """
    Divide um texto em pedaços menores usando um RecursiveCharacterTextSplitter.
    
    Args:
        texto (str): O texto a ser dividido.
    
    Returns:
        list: Uma lista de pedaços menores de texto.
    """
    separador_de_texto = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separator="\n")
    trechos = separador_de_texto.split_documents(texto)
    return trechos

def carrega_vector_db(trechos, index_name):
    """
    Carrega os embeddings para os trechos de texto fornecidos e cria um armazenamento de vetor FAISS.

        Args:
            trechos (list): Uma lista de trechos de texto.

        Returns:
            FAISS: Um armazenamento de vetor FAISS contendo os embeddings para os trechos de texto fornecidos.
    """
    embeddings = define_embedder()
    #metadata = [{"source": str(i)} for i in range(len(trechos))]
    #vector_store = FAISS.from_documents(trechos, embeddings, metadatas=metadata)
    vector_store = FAISS.from_documents(trechos, embeddings)
    vector_store.save_local(index_name)
    return vector_store

def pega_resposta(query, docs):
    """
    Usa uma cadeia de QA Langchain para encontrar a melhor resposta para uma determinada consulta.

    Args:
        query (str): A consulta a ser respondida.
        store (FAISS): Um armazenamento de vetor FAISS contendo os embeddings para os trechos de texto.

    Returns:
        dict: Um dicionário contendo a resposta à consulta dividida em duas partes: parte1 e parte2.
    """
    llm = AzureOpenAI(model_kwargs={'engine':os.environ["MODEL_DEPLOYMENT_ID"]}, model='gpt-35-turbo', client=any)
    chain = load_qa_with_sources_chain(llm=llm, chain_type="stuff")
    response = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
    response_parts = response['output_text'].split("SOURCES:")
    parte1 = response_parts[0]
    parte2 = response_parts[1] if len(response_parts) > 1 else ""
    return dict(response=response, parte1=parte1, parte2=parte2)

def conversa(vectorstore):
    """
    Cria uma cadeia de recuperação conversacional usando o armazenamento de vetor fornecido.

    Args:
        vectorstore (FAISS): Um armazenamento de vetor FAISS contendo os embeddings para os trechos de texto.

    Returns:
        ConversationalRetrievalChain: Uma cadeia de recuperação conversacional.
    """
    llm = AzureOpenAI(model_kwargs={'engine':st.session_state.modelo}, client=any)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    st.session_state['messages'].append({"role": "assistant", "content": conversation_chain})
    return conversation_chain

def gera_conversa(pergunta):
    """
    Gera uma resposta de conversa para uma determinada pergunta usando a cadeia de recuperação conversacional Langchain.

    Args:
        pergunta (str): A pergunta a ser respondida.

    Returns:
        None
    """
    response = st.session_state.conversation({'question': pergunta})
    st.session_state.chat_history = response['chat_history']
    return response

def get_text():
    """
    Obtém o texto de entrada do usuário de um widget de entrada de texto do Streamlit.

    Returns:
        str: O texto de entrada do usuário.
    """
    with st.form(key="user_input_form", clear_on_submit=True):
        st.markdown("<h1 style='text-align: center; color: #000000;'>Assistente de Pesquisas 📚</h1>", unsafe_allow_html=True)
        user_input = st.text_input(label="Caixa de texto", label_visibility="hidden" , placeholder="Sobre o que você quer falar?", key="user_input")
        submit_button = st.form_submit_button(label="Enviar")
        return user_input

def inicializa_ui():
    """
    Inicializa as variáveis de estado do Streamlit.
    """
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Olá, sou seu assistente de pesquisas 👋"]
    if 'past' not in st.session_state:
        st.session_state['past'] = ["Olá 👋"]
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
    if 'modelo' not in st.session_state:
        st.session_state['modelo'] = []
    if 'custo' not in st.session_state:
        st.session_state['custo'] = []
    if 'total_de_tokens' not in st.session_state:
        st.session_state['total_de_tokens'] = []
    if 'custo_total' not in st.session_state:
        st.session_state['custo_total'] = 0.0
    if 'modelo' not in st.session_state:
        st.session_state['modelo'] = []
    if 'user_input' not in st.session_state:
        st.session_state['user_input'] = ""

def reseta_ui():
    """
    Resets the Streamlit session state variables to their initial values.
    """
    st.session_state['generated'] = ["Olá, sou seu assistente de pesquisas 👋"]
    st.session_state['past'] = ["Olá 👋"]
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    st.session_state['modelo'] = []
    st.session_state['custo'] = []
    st.session_state['total_de_tokens'] = []
    st.session_state['custo_total'] = 0.0
    st.session_state['modelo'] = []
    st.session_state['user_input'] = []

def carrega_urls(url_list):
    """
    Carrega as URLs da lista fornecida usando o UnstructuredHtmlLoader do LangChain e divide cada documento usando o CharacterTextSplitter do LangChain.

    Args:
    url_list (list): Uma lista de URLs para carregar e dividir.

    Returns:
    list: Uma lista de documentos, onde cada documento é uma lista de trechos de texto.
    """
    loader = UnstructuredURLLoader(urls=url_list)
    splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separator="\n")
    doc = loader.load()
    data = splitter.split_documents(doc)
    return data