
"""Imports the required modules for the Streamlit web app."""
import os
import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from htmlTemplate import css
from modules.utils import carregar_credenciais
from modules.utils import processar_documentos
from modules.utils import separar_texto
from modules.utils import carrega_vector_db
from modules.utils import criar_chain_instance
from modules.utils import gerar_resposta
from modules.utils import capturar_input_usuario
from modules.utils import inicializar_ui
from modules.utils import resetar_ui
from modules.utils import carregar_base_de_conhecimento_sre
from modules.utils import limpar_uploads
from modules.utils import pesquisar_kb_sre
from modules.utils import agente


def main():
    """
    A função principal que executa o aplicativo da web Streamlit. Ele carrega os módulos necessários, define a configuração da página e processa os documentos enviados pelo usuário. Ele também permite que o usuário pesquise informações nos documentos processados.

    Retorna:
        None
    """
    carregar_credenciais()
    inicializar_ui()

    st.set_page_config(page_title="Assistente de Pesquisas",page_icon=":books:")
    st.write(css,unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    col1, col2 = st.columns([2, 4])

    col1.subheader("")
    with st.sidebar:
        #Seção para carregar documentos adicionais para análise
        st.subheader("📖 Documentos Adicionais")
        pdf_docs = st.file_uploader(label="Carregue documentos adicionais", accept_multiple_files=True, type=["pdf"], label_visibility="visible")
        if st.button("Processar"):
            with st.spinner("Processando..."):
                upload_dir = os.path.join("uploads")
                for uploaded_files in pdf_docs:
                    with open(os.path.join(upload_dir, uploaded_files.name), "wb") as f:
                        saved_file = f.write(uploaded_files.getbuffer())
                if saved_file:
                    documentos = processar_documentos(upload_dir)
                    trechos = separar_texto(documentos)
                    store = carrega_vector_db(trechos, "faiss_uploaded_docs")
                    st.session_state.conversation = criar_chain_instance(store)
                    st.success("Processamento concluído!")
                else:
                    st.error("Nenhum documento processado!")
        # Fim da seção para carregar documentos adicionais para análise
        
        # Seção para carregar bases de conhecimento conhecidas
        st.subheader("🪣 Base de Dados Fundamentais")
        
        opcoes = ["Site Reliability Engineering", "Observability [TBD]", "DevOps [TBD]"]
        opcao_selecionada = st.selectbox("Selecione uma opção", opcoes)
        if st.button("Carregar"):
            with st.spinner("Processando..."):
                t = carregar_base_de_conhecimento_sre()
                if t:
                    st.success(f"Base de conhecimento sobre {opcao_selecionada} carregada!")
        # Fim da seção para carregar bases de conhecimento conhecidas
        
        #Seção para selecionar o modelo de conversação
        
        st.subheader("🤖 Escolha seu modelo LLM")
        modelo = st.sidebar.radio("Qual modelo quer usar?:", ("GPT-3", "GPT-3.5"))
        
        if modelo == "GPT-3":
            st.session_state.modelo = "text-davinci-003"
        else:
            st.session_state.modelo = "trouble-buddy"
        
        limpar_conversa = st.sidebar.button("Limpar Conversa", key="limpar")
        if limpar_conversa:
            resetar_ui()
            limpar_uploads()
        # Fim da seção para selecionar o modelo de conversação

    col2.subheader("")
    container_pergunta = st.container()
    colored_header(label='', description='', color_name='blue-30')
    container_resposta = st.container()

    with container_pergunta:
        input_usuario = capturar_input_usuario()


    with container_resposta:
        if input_usuario:
            response = agente(input_usuario)
            #response = gerar_resposta(input_usuario)
            sre_kb_resposta = agente(input_usuario)
            if sre_kb_resposta:
                print(sre_kb_resposta['output'])
            st.session_state.past.append(input_usuario)
            st.session_state.generated.append(response['output'])
        
        if st.session_state['generated']:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
                message(st.session_state["generated"][i], avatar_style="bottts", seed="Snickers", key=str(i))



if __name__ == '__main__':
    main()
