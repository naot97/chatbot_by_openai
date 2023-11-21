import gradio as gr
import random
import time
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage
from utils_openai import usage_token_from_messages
from time import time
import pandas as pd
import os


db =  FAISS.load_local("presight_faiss", OpenAIEmbeddings(), 'presight')
chat = ChatOpenAI(request_timeout=5)

def inference(question: str, reference=None):
    begin = time()
    context  = db.similarity_search(question)[0].page_content
    search_time =  time() - begin
    print("search time: ", round( search_time, 2))

    messages = [
        SystemMessage(content="You are an assistant for question-answering tasks. Use the context to answer the question. If you don't know the answer, just say that you don't know. Use two sentences maximum and keep the answer concise"),
        HumanMessage(content=f"#Context: {context}\n\n#Question: {question}")
    ]

    usage = usage_token_from_messages(messages)
    answer = chat(messages).content
    answering_time = time() - begin - search_time
    print("answering time: ", round( answering_time, 2))

    return {
        'context': context,
        'question': question,
        'answer': answer,
        'reference':reference,
        'search_time': search_time,
        'usage': usage,
        'answering_time': answering_time,
    }

if __name__ == "__main__":
    ds = pd.read_csv("qa_presight.csv").to_dict('records')

    result = []

    for row in ds:
        result.append(inference(row["question"], row["answer"]))

    pd.DataFrame(result).to_csv("result.csv")