import os
import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.chat_models import AzureChatOpenAI as AzureOpenAI
from langchain import LLMChain
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.agents import Tool, initialize_agent, AgentExecutor, ZeroShotAgent
from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.agents.agent_toolkits import (
    create_vectorstore_router_agent,
    VectorStoreRouterToolkit,
    VectorStoreInfo,
)

from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)



def carregar_credenciais():
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

def definir_embedder():
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

def processar_documentos(upload_dir):
    """
    Extrai o texto de arquivos PDF e os concatena em uma única string.

    Args:
        pdfs (list): Uma lista de arquivos PDF.

    Returns:
        str: Uma string contendo o texto concatenado de todos os arquivos PDF.
    """
    #texto = ""
    #for pdf in pdfs:
    #    pdf_reader = PdfReader(pdf)
    #    for pages in pdf_reader.pages:
    #        texto += pages.extract_text()
    #return texto
    loader = PyPDFDirectoryLoader(upload_dir)
    texto = loader.load()
    return texto

def separar_texto(documentos):
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
    trechos = separador_de_texto.split_documents(documentos)
    return trechos

def carregar_vector_db(trechos, index_name):
    """
    Carrega os embeddings para os trechos de texto fornecidos e cria um armazenamento de vetor FAISS.

        Args:
            trechos (list): Uma lista de trechos de texto.

        Returns:
            FAISS: Um armazenamento de vetor FAISS contendo os embeddings para os trechos de texto fornecidos.
    """
    embeddings = definir_embedder()
    #metadata = [{"source": str(i)} for i in range(len(trechos))]
    #vector_store = FAISS.from_documents(trechos, embeddings, metadatas=metadata)
    vector_store = FAISS.from_documents(trechos, embeddings)
    vector_store.save_local(index_name)
    criar_vectorstore_session(index_name)
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

def criar_chain_instance(vectorstore):
    """
    Cria uma cadeia de recuperação conversacional usando o armazenamento de vetor fornecido.

    Args:
        vectorstore (FAISS): Um armazenamento de vetor FAISS contendo os embeddings para os trechos de texto.

    Returns:
        ConversationalRetrievalChain: Uma cadeia de recuperação conversacional.
    """
    #llm = AzureOpenAI(model_kwargs={'engine':st.session_state.modelo}, client=any, temperature=0.0)
    llm = AzureOpenAI(deployment_name='trouble-buddy', model_name='gpt-3.5-turbo', temperature=0.0, client=any)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    st.session_state['messages'].append({"role": "assistant", "content": conversation_chain})
    return conversation_chain

def gerar_resposta(input_usuario):
    """
    Gera uma resposta de conversa para uma determinada pergunta usando a cadeia de recuperação conversacional Langchain.

    Args:
        pergunta (str): A pergunta a ser respondida.

    Returns:
        None
    """
    resposta = st.session_state.conversation({'question': input_usuario})
    st.session_state.chat_history = resposta['chat_history']
    return resposta

def capturar_input_usuario():
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

def inicializar_ui():
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
    if 'vectorstore' not in st.session_state:
        st.session_state['vectorstore'] = ""
    limpar_uploads()

def resetar_ui():
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
    st.session_state['user_input'] = ""
    st.session_state['vectorstore'] = ""

    limpar_uploads()

def carregar_urls(url_list):
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

def sre_building_secure_and_reliable_systems():
    """
    Carrega as URLs do livro Building Secure and Reliable Systems do Google e as processa em um banco de vetores.

    Retorna:
    Nenhum
    """
    url_list = []
    for i in range(1,22):
        if i < 10:
            url_list.append("https://google.github.io/building-secure-and-reliable-systems/raw/ch0" + str(i) + ".html")
        else:
            url_list.append("https://google.github.io/building-secure-and-reliable-systems/raw/ch" + str(i) + ".html")

    print(url_list)
    docs = carregar_urls(url_list)
    print(docs)
    vector_urls = carregar_vector_db(docs, "sre_building_secure_and_reliable_systems")
    if vector_urls:
        return vector_urls

def sre_site_reliability_engineering():
    """
    Carrega as URLs do livro Site Reliability Engineering do Google e as processa em um banco de vetores.

    Retorna
    url
    """

    url_list = [
        "https://sre.google/sre-book/foreword/",
        "https://sre.google/sre-book/preface/",
        "https://sre.google/sre-book/part-I-introduction/",
        "https://sre.google/sre-book/introduction/",
        "https://sre.google/sre-book/production-environment/",
        "https://sre.google/sre-book/part-II-principles/",
        "https://sre.google/sre-book/embracing-risk/",
        "https://sre.google/sre-book/service-level-objectives/",
        "https://sre.google/sre-book/eliminating-toil/",
        "https://sre.google/sre-book/monitoring-distributed-systems/",
        "https://sre.google/sre-book/automation-at-google/",
        "https://sre.google/sre-book/release-engineering/",
        "https://sre.google/sre-book/simplicity/",
        "https://sre.google/sre-book/part-III-practices/",
        "https://sre.google/sre-book/practical-alerting/",
        "https://sre.google/sre-book/being-on-call/",
        "https://sre.google/sre-book/effective-troubleshooting/",
        "https://sre.google/sre-book/emergency-response/",
        "https://sre.google/sre-book/managing-incidents/",
        "https://sre.google/sre-book/postmortem-culture/",
        "https://sre.google/sre-book/tracking-outages/",
        "https://sre.google/sre-book/testing-reliability/",
        "https://sre.google/sre-book/software-engineering-in-sre/",
        "https://sre.google/sre-book/load-balancing-frontend/",
        "https://sre.google/sre-book/load-balancing-datacenter/",
        "https://sre.google/sre-book/handling-overload/",
        "https://sre.google/sre-book/addressing-cascading-failures/",
        "https://sre.google/sre-book/managing-critical-state/",
        "https://sre.google/sre-book/distributed-periodic-scheduling/",
        "https://sre.google/sre-book/data-processing-pipelines/",
        "https://sre.google/sre-book/data-integrity/",
        "https://sre.google/sre-book/reliable-product-launches/",
        "https://sre.google/sre-book/part-IV-management/",
        "https://sre.google/sre-book/accelerating-sre-on-call/",
        "https://sre.google/sre-book/dealing-with-interrupts/",
        "https://sre.google/sre-book/operational-overload/",
        "https://sre.google/sre-book/communication-and-collaboration/",
        "https://sre.google/sre-book/evolving-sre-engagement-model/",
        "https://sre.google/sre-book/part-V-conclusions/",
        "https://sre.google/sre-book/lessons-learned/",
        "https://sre.google/sre-book/conclusion/",
        "https://sre.google/sre-book/availability-table/",
        "https://sre.google/sre-book/service-best-practices/",
        "https://sre.google/sre-book/incident-document/",
        "https://sre.google/sre-book/example-postmortem/",
        "https://sre.google/sre-book/launch-checklist/",
        "https://sre.google/sre-book/production-meeting/",
        "https://sre.google/sre-book/bibliography/",
        "https://sre.google/sre-book/table-of-contents/",
        "https://sre.google/sre-book/foreword/",
        "https://sre.google/sre-book/preface/",
        "https://sre.google/sre-book/part-I-introduction/",
        "https://sre.google/sre-book/introduction/",
        "https://sre.google/sre-book/production-environment/",
        "https://sre.google/sre-book/part-II-principles/",
        "https://sre.google/sre-book/embracing-risk/",
        "https://sre.google/sre-book/service-level-objectives/",
        "https://sre.google/sre-book/eliminating-toil/",
        "https://sre.google/sre-book/monitoring-distributed-systems/",
        "https://sre.google/sre-book/automation-at-google/",
        "https://sre.google/sre-book/release-engineering/",
        "https://sre.google/sre-book/simplicity/",
        "https://sre.google/sre-book/part-III-practices/",
        "https://sre.google/sre-book/practical-alerting/",
        "https://sre.google/sre-book/being-on-call/",
        "https://sre.google/sre-book/effective-troubleshooting/",
        "https://sre.google/sre-book/emergency-response/",
        "https://sre.google/sre-book/managing-incidents/",
        "https://sre.google/sre-book/postmortem-culture/",
        "https://sre.google/sre-book/tracking-outages/",
        "https://sre.google/sre-book/testing-reliability/",
        "https://sre.google/sre-book/software-engineering-in-sre/",
        "https://sre.google/sre-book/load-balancing-frontend/",
        "https://sre.google/sre-book/load-balancing-datacenter/",
        "https://sre.google/sre-book/handling-overload/",
        "https://sre.google/sre-book/addressing-cascading-failures/",
        "https://sre.google/sre-book/managing-critical-state/",
        "https://sre.google/sre-book/distributed-periodic-scheduling/",
        "https://sre.google/sre-book/data-processing-pipelines/",
        "https://sre.google/sre-book/data-integrity/",
        "https://sre.google/sre-book/reliable-product-launches/",
        "https://sre.google/sre-book/part-IV-management/",
        "https://sre.google/sre-book/accelerating-sre-on-call/",
        "https://sre.google/sre-book/dealing-with-interrupts/",
        "https://sre.google/sre-book/operational-overload/",
        "https://sre.google/sre-book/communication-and-collaboration/",
        "https://sre.google/sre-book/evolving-sre-engagement-model/",
        "https://sre.google/sre-book/part-V-conclusions/",
        "https://sre.google/sre-book/lessons-learned/",
        "https://sre.google/sre-book/conclusion/",
        "https://sre.google/sre-book/availability-table/",
        "https://sre.google/sre-book/service-best-practices/",
        "https://sre.google/sre-book/incident-document/",
        "https://sre.google/sre-book/example-postmortem/",
        "https://sre.google/sre-book/launch-checklist/",
        "https://sre.google/sre-book/production-meeting/",
        "https://sre.google/sre-book/bibliography/"
    ]
    
    print(url_list)
    docs = carregar_urls(url_list)
    print(docs)
    vector_urls = carregar_vector_db(docs, "site_reliability_engineering")
    if vector_urls:
        return vector_urls

def limpar_uploads():
    """
    Exclui todos os arquivos com sufixo pdf no diretório de uploads.

    Args:
    Nenhum

    Returns:
    Nenhum
    """

    pdf_files = [f for f in os.listdir("uploads") if f.endswith(".pdf")]
    for filename in pdf_files:
        file_path = os.path.join("uploads", filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Falha ao excluir {file_path}. Motivo: {e}")
        file_path = os.path.join("uploads", filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Falha ao excluir {file_path}. Motivo: {e}")

def criar_vectorstore_session(index_name):
    """
    Cria uma sessão VectorStore para um determinado índice.

    Args:
    index_name (str): O nome do índice para criar uma sessão VectorStore.

    Returns:
    VectorStoreSession: Uma sessão VectorStore para o índice fornecido.
    """
    if index_name not in st.session_state:
        st.session_state[index_name] = index_name


    """
    Pesquisa a base de conhecimento do livro Building Secure and Reliable Systems do Google para encontrar as 3 respostas mais similares à entrada do usuário.

    Args:
    input_usuario (str): A entrada do usuário a ser pesquisada na base de conhecimento.

    Returns:
    list: Uma lista contendo as 3 respostas mais similares à entrada do usuário.
    """

def pesquisar_kb_brds(input_usuario):
    embeddings = definir_embedder()
    kb_brds = FAISS.load_local("sre_building_secure_and_reliable_systems", embeddings)
    llm = AzureOpenAI(deployment_name='trouble-buddy', model_name='gpt-3.5-turbo', temperature=0.0, client=any)
    #memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=kb_brds.as_retriever(),
        #memory=memory
    )

    resposta = qa.run(input_usuario)
    return resposta

def pesquisar_documentos_upload(input_usuario):
    embeddings = definir_embedder()
    uploaded_docs = FAISS.load_local("faiss_uploaded_docs", embeddings)
    llm = AzureOpenAI(deployment_name='trouble-buddy', model_name='gpt-3.5-turbo', temperature=0.0, client=any)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    
    #qa = RetrievalQA.from_chain_type(
    #    llm=llm,
    #    retriever=uploaded_docs.as_retriever()
    #    #memory=memory
    #)
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        uploaded_docs.as_retriever(),
        memory=memory
    )


    resposta = qa.run(input_usuario)
    return resposta

def pesquisar_kb_sre(input_usuario):
    embeddings = definir_embedder()
    kb_sre = FAISS.load_local("site_reliability_engineering", embeddings)
    llm = AzureOpenAI(deployment_name='trouble-buddy', model_name='gpt-3.5-turbo', temperature=0.0, client=any)
    #memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=kb_sre.as_retriever(),
        #memory=memory
    )

    resposta = qa.run(input_usuario)
    return resposta

def agente(input_usuario):
    """
    Esta função serve como o ponto de entrada principal para o agente do chatbot. Ela recebe uma string de entrada do usuário e retorna uma string de resposta gerada pelo agente.

    Args:
        input_usuario (str): A string de entrada do usuário.

    Returns:
        str: A string de resposta gerada pelo agente.
    """


    embeddings = definir_embedder()

    llm = AzureOpenAI(
        deployment_name='trouble-buddy', 
        model_name='gpt-3.5-turbo', 
        temperature=0.0, 
        client=any,
        )
    
    """
    Tookit para o agente do chatbot.
    """
    uploaded_docs = FAISS.load_local("faiss_uploaded_docs", embeddings)
    upload_vectorstore_info = VectorStoreInfo(
        name="Documentos enviados",
        description="Informações sobre os documentos enviados pelo usuário",
        vectorstore=uploaded_docs,
    )

    kb_sre = FAISS.load_local("site_reliability_engineering", embeddings)
    sre_vectorstore_info = VectorStoreInfo(
        name="Base de Conhecimento sobre o livro Site Reliability Engineering do Google",
        description="Informações sobre o livro Site Reliability Engineering do Google",
        vectorstore=kb_sre,
    )

    kb_brds = FAISS.load_local("sre_building_secure_and_reliable_systems", embeddings)
    brds_vectorstore_info = VectorStoreInfo(
        name="Based de Conhecimento sobre o livro Building Secure and Reliable Systems do Google",
        description="Informações sobre o livro Building Secure and Reliable Systems do Google",
        vectorstore=kb_brds,
    )    

    router_toolkit = VectorStoreRouterToolkit(
    vectorstores=[upload_vectorstore_info, sre_vectorstore_info, brds_vectorstore_info], llm=llm
    )

    prefix = """
    prefix: str = 'You are an agent designed to answer questions.
    You have access to tools for interacting with different sources, and the inputs to the tools are questions.
    Your main task is to decide which of the tools is relevant for answering question at hand.
    For complex questions, you can break the question down into sub questions and use tools to answers the sub questions.
    You always answer in portuguese.
    """

    agent_executor = create_vectorstore_router_agent(
        llm=llm, 
        toolkit=router_toolkit, 
        verbose=True,
        prefix=prefix,
        )
    
    d = agent_executor.run(input_usuario)
    return d