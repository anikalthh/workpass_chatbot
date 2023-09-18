# standard library modules
import os
import re

# third-party modules
import streamlit as st
from streamlit_chat import message
from dotenv import (
    find_dotenv,
    load_dotenv
)
from deep_translator import GoogleTranslator, single_detection

# SUPPORTED LANGUAGES
# langs_list = GoogleTranslator().get_supported_languages() 
# print("-------------------LANGUAGES--------------------")
# print(langs_list)

# local modules
from function import (
    conversational_chat,
    start_conversation,
    start_conversation_refine,
    goodFeedback,
    badFeedback
)

from constants import (
    TRANSLATOR_API_KEY
)

load_dotenv(find_dotenv())

# Default text
generated_session_text = "Hello! I'm your guide for migrant domestic workers. Ask me anything!"
past_session_text = "Hey! 👋"
welcome_text = "How would you like us to help you today?"
button_text = "Send"

session_state_default = {
    'history': [],
    'generated': [generated_session_text],
    'past': [past_session_text],
    'queryid': [],
    'resultids': [],
}

for name, value in session_state_default.items():
    st.session_state[name] = value

chain = start_conversation()

# container for the chat history
response_container = st.container()

# container for the user's text input
container = st.container()

with container:
    with st.form(key='sgwp', clear_on_submit=True):

        user_input = st.text_input(
            welcome_text,
            max_chars=200
        )
        send_button = st.form_submit_button(label=button_text)

        with st.spinner('loading...'):
            if send_button and user_input:
                lang = single_detection(user_input, api_key=TRANSLATOR_API_KEY)
                print("------------DETECTED LANG------------:", lang)
                print("---------------ORIGINAL--------------:", user_input)
                
                if lang != 'en':
                    translated = GoogleTranslator(source='auto', target='en').translate(user_input)
                    print("-------------TRANSLATED--------------:", translated)
                    response_text = translated
                    
                else:
                    response_text = user_input

                response_text = "As a migrant domestic worker, " + response_text 
                output = conversational_chat(
                    chain,
                    response_text
                )

                st.session_state['past'].append(user_input)
                output = output.replace("$", "SGD")

                if lang in ['sk', 'ceb']:
                    lang = 'tl'

                og = GoogleTranslator(source='en', target=lang).translate(output)
                st.session_state['generated'].append(og)

    if st.button('Reset this conversation?'):
        for name, value in session_state_default.items():
            st.session_state[name] = value


if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="personas")
            message(st.session_state["generated"][i], key=str(i), avatar_style="bottts")

            if (st.session_state['generated'][i] != generated_session_text):
                col1, col2, col3, col4 = st.columns([2, 1, 1, 14])

                with col2:
                    st.button('👍', key=st.session_state['queryid'][i-1]+"a", on_click=goodFeedback, args=(st.session_state['queryid'][i-1], st.session_state['resultids'][i-1]))
                with col3:
                    st.button('👎', key=st.session_state['queryid'][i-1]+"b", on_click=badFeedback, args=(st.session_state['queryid'][i-1], st.session_state['resultids'][i-1]))
                    
