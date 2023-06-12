# Streamlit Chatbot

Este é um chatbot que implementa o padrão de 'Augmented Retrieval' para recuperar informações em fontes de dados privadas usando o gpt-3.5-turbo (ChatGPT) é construído usando o Streamlit, uma estrutura de aplicativo da web Python que permite criar aplicativos da web interativos com facilidade.

# Configurando o ambiente

## Assegure que você tem a versão correta do Python instalada
|:exclamation: NOTA IMPORTANTE|
|-----------------------------|

**Warning**

 A versão do Python utilizada nesse projecto está declarada no arquivo runtime.txt. Assegure-se de ter a mesma versão instalada e que ela é a versão 'linkada' ao comando 'python'

```bash
$ python --version 
Python 3.10.4
```

## Crie um ambiente virtual (venv)
```bash
python -m venv .venv
```
## Ative o ambiente virtual
<details>

<summary>Linux version</summary>

```bash
source .venv/bin/activate
```
</details>
<details>
<summary>Windows Version</summary>
In cmd.exe

```cmd
venv\Scripts\activate.bat
```
In PowerShell
```Powershell
venv\Scripts\Activate.ps1
```
</details>

## Instalação
Para instalar as dependências do projeto, execute o seguinte comando:

```
pip install -r requirements.txt
```

## Uso
Para executar o aplicativo da web, execute o seguinte comando:

```
streamlit run copilot.py
```

Isso iniciará o aplicativo da web e abrirá uma nova janela do navegador. Você pode então usar o chatbot para pesquisar informações nos documentos fornecidos.

## Base de Dados Fundamentais
O código já tem scrappers para criar 2 vectorDBs (FAISS) baseados em informações públicas. São elas:

- Site Reliability Engineering (https://sre.google/sre-book/table-of-contents/)
- Building Secure & Reliable Systems (https://google.github.io/building-secure-and-reliable-systems/raw/toc.html)

## Contribuindo
Se você quiser contribuir para este projeto, sinta-se à vontade para enviar um pull request. Certifique-se de seguir as diretrizes de contribuição e de teste.

## Licença
Este projeto é licenciado sob a licença MIT. Consulte o arquivo LICENSE para obter mais informações.

## Créditos
Este projeto foi criado por Ricardo Coelho de Sousa (rcsousa) para comunidade #SREBrasil.