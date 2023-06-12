# Streamlit Chatbot

Este é um chatbot que implementa o padrão de 'Augmented Retrieval' para recuperar informações em fontes de dados privadas usando o gpt-3.5-turbo (ChatGPT) é construído usando o Streamlit, uma estrutura de aplicativo da web Python que permite criar aplicativos da web interativos com facilidade.

# Clonando o Repo

A partir do prompt to seu sistema operacional
```
$ git clone https://github.com/rcsousa/LangChainLab.git
```

# Configurando o ambiente

## Assegure que você tem a versão correta do Python instalada
|:exclamation: NOTA IMPORTANTE|
|-----------------------------|

 A versão do Python utilizada nesse projecto está declarada no arquivo runtime.txt. Assegure-se de ter a mesma versão instalada e que ela é a versão 'linkada' ao comando 'python'

```bash
$ python --version 
Python 3.10.4
```

## Crie um ambiente virtual (venv)
```Shell
$ python -m venv .venv
```
## Ative o ambiente virtual
<details>

<summary>Linux version</summary>

```Shell
$ source .venv/bin/activate
```
</details>
<details>
<summary>Windows Version</summary>
No cmd.exe

```Shell
c:\venv\Scripts\activate.bat
```
No PowerShell
```Shell
PS C:\venv\Scripts\Activate.ps1
```
</details>

# Instalando as dependências
Para instalar as dependências do projeto, execute o seguinte comando:

```Shell
pip install -r requirements.txt
```

# Iniciando o chatbot
Para executar o aplicativo da web, execute o seguinte comando:

```Shell
streamlit run copilot.py
```

Isso iniciará o aplicativo da web e abrirá uma nova janela do navegador. Você pode então usar o chatbot para pesquisar informações nos documentos fornecidos.

# Base de Dados Fundamentais
O código já tem scrappers para criar 2 vectorDBs (FAISS) baseados em informações públicas. São elas:

- Site Reliability Engineering (https://sre.google/sre-book/table-of-contents/)
- Building Secure & Reliable Systems (https://google.github.io/building-secure-and-reliable-systems/raw/toc.html)

:warning: **Lentidão ao carregar as bases fundamentais**

```text
Dependendo da velocidade da rede e do rate limit para uso das APIs de embedding o processo de carregamento das bases fundamentais pode demorar uns 2 minutos.
```


## Contribuindo
Se você quiser contribuir para este projeto, sinta-se à vontade para enviar um pull request. Certifique-se de seguir as diretrizes de contribuição e de teste.

## Licença
Este projeto é licenciado sob a licença MIT. Consulte o arquivo LICENSE para obter mais informações.

## Créditos
Este projeto foi criado por Ricardo Coelho de Sousa (rcsousa) para comunidade #SREBrasil.