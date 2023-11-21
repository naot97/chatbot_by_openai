import gradio as gr
import random
import time
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage
from time import time
import os


db =  FAISS.load_local("presight_faiss", OpenAIEmbeddings(), 'presight')
chat = ChatOpenAI(request_timeout=5)
def init_prompt():
    global messages
    messages = [
        SystemMessage(content="You are an assistant for question-answering tasks. Use the context to answer the question. If you don't know the answer, just say that you don't know. Use two sentences maximum and keep the answer concise"),
    ]
init_prompt()

with gr.Blocks() as demo:
    # gr.image(["presight.png"], width=120)
    chatbot = gr.Chatbot(value=[[None, "👋 Hello! Welcome to Presight's Chatbot.\nIf you have any questions regarding our Privacy Policy or data protection practices, feel free to ask."]])
    msg = gr.Textbox("Do your company protect the privacy of the customers?")
    clear = gr.ClearButton([msg, chatbot])

    def respond(question, chat_history):
        begin = time()
        context  = db.similarity_search(question)[0].page_content
        print("search time: ", round( time() - begin, 2))
        # relevance_score, relevance_doc = db.similarity_search_with_score(question)
        # print(relevance_score, relevance_doc) 
        messages.append(HumanMessage(content=f"#Context: {context}\n\n#Question: {question}"))
        # print(messages)
        bot_message = chat(messages)
        messages.append(bot_message)
        print("response time: ", round( time() - begin, 2))
        chat_history.append((question, bot_message.content))
        return "", chat_history


    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(init_prompt)
    
if __name__ == "__main__":
    demo.launch()
