# Chatbot Application with Streamlit Chat

This is a chatbot application built using Streamlit Chat that utilises several dependencies to provide conversational functionality. 
The application is designed to allow users to interact with a chatbot that can answer questions and provide information by querying an AWS Kendra vector index.

---

# Requirements
To run the app, you will need to have the following installed:

1. Python 3.6 or higher
2. Virtualenv

---

# Dependencies

1. **langchain**: Framework for LLMs.
2. **langdetect**: Python library for detecting the language of a given text.
3. **openai**: Python library for accessing OpenAI's GPT models.
4. **python-dotenv**: Python library for loading environment variables from a .env file.
5. **streamlit**: Python library for building interactive web applications.
6. **streamlit_chat**: Streamlit component that provides a chat widget for use in web applications.
7. **tiktoken**: Fast BPE tokeniser for use with OpenAI's models. (Not from TikTok)
8. **aws kendra**: Vector indexer with semantic search capabilities. 

---

# Installation
1. Clone the repository
```bash
$ git clone https://github.com/anikalthh/workpass_chatbot.git
```
2. Create a virtual environment:
```bash
$ python -m venv .venv
$ source .venv/bin/activate
```
3. Install the dependencies:
```bash
(venv)$ pip install -r requirements.txt
```

---

# Environment Variables

| KEY | VALUE |
|---|---|
| OPENAI_API_KEY | string |
| KENDRA_INDEX_ID | string |
| AWS_ACCESS_KEY_ID | string |
| AWS_SECRET_ACCESS_KEY | string |
| AWS_DEFAULT_REGION | string e.g. "us-east-1" |
| TRANSLATION_API_KEY | string |

TRANSLATION_API_KEY can be generated on https://detectlanguage.com/documentation

# Usage

## AWS Kendra Index
1. Set up your AWS account and create an index on AWS Kendra
2. Connect your data source and let Kendra perform the indexing

## Environment Set-Up
1. Create a `.env` file and add your various environment variables.
2. If you are hosting locally, comment out the constants which are obtained from streamlit (e.g. OPENAI_API_KEY == st.secrets["OPENAI_API_KEY"])
3. If you are hosting on streamlit, include all the enviroment variables in the secrets folder on streamlit.

---

# Folder Structure
```bash
├── LICENSE
├── README.md
├── app
│   ├── __init__.py
│   ├── constants.py
│   ├── function.py
│   ├── main.py
│   └── static
└── requirements.txt
```

---

# Local Testing
Execute the following command to test the chatbot locally.

```bash
(venv)$ streamlit run app/main.py
```
There is automated code in the script that logs all the testing questions asked in the chatbot into CSV files in the app/prev_records folder.

---

# References
https://github.com/Papagoat/llm-sgwp-chatbot.git

https://github.com/aws-samples/amazon-kendra-langchain-extensions/blob/main/kendra_retriever_samples/kendra_chat_open_ai.py

---

# Contributing
Contributions are welcome! If you find a bug or want to suggest a new feature, please open an issue or submit a pull request.

---

# License
This project is licensed under the MIT License.
