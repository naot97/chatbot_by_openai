import gradio as gr
import random
import time
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage
from time import time
import os


db =  FAISS.load_local("demo", OpenAIEmbeddings())
chat = ChatOpenAI(request_timeout=5)
def init_prompt():
    global messages
    messages = [
        SystemMessage(content="You are a help full chatbot. You will be provided a part of transcript as a context. Please answer the question based on the context."),
    ]
init_prompt()

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(value=[[None, "👋 Hello! Welcome to Eklipse's Chatbot.\nIf you have any questions regarding '`Survive 100 Days Trapped, Win $500,000`' video, feel free to ask."]])
    msg = gr.Textbox("What is the prize?")
    clear = gr.ClearButton([msg, chatbot])

    def respond(question, chat_history):
        begin = time()
        context  = db.similarity_search(question)[0].page_content
        print("search time: ", round( time() - begin, 2))
        messages.append(HumanMessage(content=f"#Context: {context}\n\n#Question: {question}"))
        bot_message = chat(messages)
        messages.append(bot_message)
        print("response time: ", round( time() - begin, 2))
        chat_history.append((question, bot_message.content))
        return "", chat_history


    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(init_prompt)
    
if __name__ == "__main__":
    demo.launch(share=True)
