# standard library modules
import os

# thid part modules
from dotenv import (
    find_dotenv,
    load_dotenv
)
import streamlit as st

load_dotenv(find_dotenv())

DOCUMENT = "Combined_MDW_OCR_16Jun23.pdf"
MODEL_NAME = 'gpt-3.5-turbo-0301'
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
TEMPERATURE = 0
KENDRA_INDEX_ID = os.getenv("KENDRA_INDEX_ID")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
TRANSLATOR_API_KEY = os.getenv("TRANSLATOR_API_KEY")

# # Commented out to run locally
# AWS_DEFAULT_REGION == st.secrets["AWS_DEFAULT_REGION"]
# AWS_SECRET_ACCESS_KEY == st.secrets["AWS_SECRET_ACCESS_KEY"]
# AWS_ACCESS_KEY_ID == st.secrets["AWS_ACCESS_KEY_ID"]
# OPENAI_API_KEY == st.secrets["OPENAI_API_KEY"]
# KENDRA_INDEX_ID == st.secrets["KENDRA_INDEX_ID"]