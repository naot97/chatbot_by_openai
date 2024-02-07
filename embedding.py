from langchain_openai import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import TokenTextSplitter

embeddings_model = OpenAIEmbeddings()

with open("tactiq-free-transcript-9RhWXPcKBI8.txt") as f:
    data = f.read()

 

text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=0)


texts = text_splitter.split_text(data)


print(len(texts))
print(texts[0])

db = FAISS.from_texts(texts, embeddings_model)


db.save_local("demo")
