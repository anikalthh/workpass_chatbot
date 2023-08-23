# standard library modules
import os
import re
import pprint
import csv
import pytz
from datetime import datetime

# third-party modules
import boto3
from botocore.config import Config

import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.retrievers import AmazonKendraRetriever
from langchain.prompts import PromptTemplate

# local modules
from constants import (
    MODEL_NAME,
    OPENAI_API_KEY,
    KENDRA_INDEX_ID,
    TEMPERATURE,
    AWS_DEFAULT_REGION
)

kendra = boto3.client('kendra')

# Build prompt
<<<<<<< Updated upstream
condense_template = """Given the following conversation and a follow up question, if they're of the same topic,
rephrase the follow up question to be a standalone question, but do not add words like "Follow up question:" or "Rephrased question:" into the rephrased questions at all. If the follow up question is a different topic, do not change the question.
=======
condense_template = """Given the following conversation and a follow up question, if they are of the same topic,
rephrase the follow up question to be a standalone question SLIGHTLY with the context of the history, but do not change more than 5 words.

If the question is not related to the previous topic, do not rephrase the question.

Additionally, do not add words like "Follow up question:" or "Rephrased question:" into the rephrased questions at all.

After generating the follow up question, review it to check if it is consistent with the above instructions or there are improvements to be made.

>>>>>>> Stashed changes
Chat History:
{chat_history}

Follow up question:
{question}
"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_template)

helpline_text = '''
I'm sorry, but I do not have the answer to your question.

You may try the following helplines:
• MDW Helpline for MDWs in distress: 1800 339 5505
• The Singapore Police Force if you are in danger: 999
• MOM Customer Service for general enquiries: 6438 5122
'''

qa_template = """You are a chatbot meant to answer queries sent by migrant domestic workers, 
<<<<<<< Updated upstream
                solely with the following context provided.
                If you don't know the answer, or if the context does not answer the original question before the rephrasing, 
                say "I'm sorry, but I do not have the answer to your question.", don't try to make up an answer.
=======
solely with the following context provided.
If you don't know the answer, or if the context does not answer the original question before the rephrasing, 
say "{helpline_text}", don't try to make up an answer.
>>>>>>> Stashed changes

                For questions with a list of answers, display the list in your response.

                Respond like you would to a migrant domestic worker.
                Context: {context}
                Question: {question}
            """

QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question", "helpline_text"],template=qa_template)

def start_conversation():

<<<<<<< Updated upstream
    retriever = AmazonKendraRetriever(index_id=KENDRA_INDEX_ID, top_k=3)
=======
    retriever = AmazonKendraRetriever(index_id=KENDRA_INDEX_ID, top_k=3, region=AWS_DEFAULT_REGION)
>>>>>>> Stashed changes
    llm=ChatOpenAI(
            temperature=TEMPERATURE,
            model_name=MODEL_NAME,
            openai_api_key=OPENAI_API_KEY,
            verbose=True
        )

    chain = ConversationalRetrievalChain.from_llm(
        combine_docs_chain_kwargs = {'prompt': QA_CHAIN_PROMPT},
        llm=llm,
        retriever=retriever,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        return_source_documents = True,
        return_generated_question = True,
        rephrase_question = False, # Does not return condensed question to LLM, only uses it for retrieval
        verbose=True)
    return chain


def conversational_chat(chain, query):
    resultIds = []
    queryId = ""

    result = chain({"question": query, "chat_history": st.session_state['history'], "helpline_text": helpline_text})
    st.session_state['history'].append((query, result["answer"]))
    output = result['answer']
    queryId = result['source_documents'][0].metadata['result_id'][:36] # The queryid is the first 36 characters of the results-id string

<<<<<<< Updated upstream
    if bool(re.search("I do not have the answer", output)) | bool(re.search("amazonaws", result['source_documents'][0].metadata['source'])):
=======
    if bool(re.search("do not have the answer to your question", output)) | bool(re.search("amazonaws", result['source_documents'][0].metadata['source'])):
>>>>>>> Stashed changes
        for d in result['source_documents']:
            resultIds.append(d.metadata['result_id'])
    else:
        output = output + '\n \n Related Source(s):'
        urls=[]
        for d in result['source_documents']:
<<<<<<< Updated upstream
            if not bool(re.search(d.metadata['source'], output)):
                output += '\n' + d.metadata['source']
=======
            if d.metadata['source'] not in urls:
                output += '\n' + d.metadata['source']
                urls.append(d.metadata['source'])
>>>>>>> Stashed changes
                resultIds.append(d.metadata['result_id'])

    st.session_state['queryid'].append(queryId)   
    st.session_state['resultids'].append(resultIds)            
    #output = result['answer'] + '\n \n Source: ' + ((result['source_documents'][0]).metadata)['source']

    # Write to CSVs
    header = ["Time_Enquired", "QueryId", "ResultIds", "Original Question", "Generated Question", "Answer", "Source_Doc", "Chat_History"]
    now = datetime.strftime(datetime.now(pytz.timezone('Asia/Singapore')), "%Y-%m-%d %H:%M:%S")
    data = [now, queryId, resultIds, query, result['generated_question'], result['answer'], result['source_documents'], st.session_state['history']]
    with open("./app/prev_records/qna.csv", 'a', encoding='UTF8',newline='') as f:
        writer = csv.writer(f)
        # writer.writerow(header)
        writer.writerow(data)

    return output

def goodFeedback(queryid, resultids):
    relevance_value = "RELEVANT"
    relevance_items = {}
    tempList = []
    for id in resultids:
        tempList.append(id)
        relevance_items = {
            "ResultId": id,
            "RelevanceValue": relevance_value,
        }

    feedback = kendra.submit_feedback(
        QueryId = queryid,
        IndexId = KENDRA_INDEX_ID,
        RelevanceFeedbackItems = [relevance_items]
    )

    # Write to CSVs
    header = ["Time_Enquired", "QueryId", "ResultIds", "Status", "Feedback"]
    now = datetime.strftime(datetime.now(pytz.timezone('Asia/Singapore')), "%Y-%m-%d %H:%M:%S")
    data = [now, queryid, tempList, "RELEVANT", feedback]
    with open("./app/prev_records/feedback.csv", 'a', encoding='UTF8',newline='') as f:
        writer = csv.writer(f)
        # writer.writerow(header)
        writer.writerow(data)

def badFeedback(queryid, resultids):
    relevance_value = "NOT_RELEVANT"
    relevance_items = {}
    tempList = []
    for id in resultids:
        tempList.append(id)
        relevance_items = {
            "ResultId": id,
            "RelevanceValue": relevance_value,
        }

    feedback = kendra.submit_feedback(
        QueryId = queryid,
        IndexId = KENDRA_INDEX_ID,
        RelevanceFeedbackItems = [relevance_items]
    )

    # Write to CSVs
    header = ["Time_Enquired", "QueryId", "ResultIds", "Status", "Feedback"]
    now = datetime.strftime(datetime.now(pytz.timezone('Asia/Singapore')), "%Y-%m-%d %H:%M:%S")
    data = [now, queryid, tempList, "NOT_RELEVANT", feedback]
    with open("./app/prev_records/feedback.csv", 'a', encoding='UTF8',newline='') as f:
        writer = csv.writer(f)
        # writer.writerow(header)
        writer.writerow(data)