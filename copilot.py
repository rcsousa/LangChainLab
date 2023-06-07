
"""Imports the required modules for the Streamlit web app."""

import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from htmlTemplate import css
from modules.utils import carrega_credenciais, processa_documentos, separa_texto, carrega_vector_db, conversa, gera_conversa, get_text, inicializa_ui, reseta_ui


def main():
    """
    A função principal que executa o aplicativo da web Streamlit. Ele carrega os módulos necessários, define a configuração da página e processa os documentos enviados pelo usuário. Ele também permite que o usuário pesquise informações nos documentos processados.

    Retorna:
        None
    """
    carrega_credenciais()
    inicializa_ui()

    st.set_page_config(page_title="Assistant de Pesquisas",page_icon=":books:")
    st.write(css,unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    col1, col2 = st.columns([2, 4])

    col1.subheader("")
    with st.sidebar:
        st.subheader("📖 Documentos")
        pdf_docs = st.file_uploader(label="Carregue seus documentos", accept_multiple_files=True, type=["pdf"])
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
        modelo = st.sidebar.radio("Escolha o modelo:", ("GPT-3", "GPT-3.5"))
        
        # Mapear modelo
        if modelo == "GPT-3":
            st.session_state.modelo = "text-davinci-003"
        else:
            st.session_state.modelo = "trouble-buddy"
        
        limpar_conversa = st.sidebar.button("Limpar Conversa", key="limpar")
        if limpar_conversa:
            reseta_ui()



    col2.subheader("")
    container_pergunta = st.container()
    colored_header(label='', description='', color_name='blue-30')
    container_resposta = st.container()

    with container_pergunta:
        user_input = get_text()


    with container_resposta:
        if user_input:
            response = gera_conversa(user_input)
            st.session_state.past.append(user_input)
            st.session_state.generated.append(response['answer'])
        
        if st.session_state['generated']:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
                message(st.session_state["generated"][i], key=str(i))



if __name__ == '__main__':
    main()
