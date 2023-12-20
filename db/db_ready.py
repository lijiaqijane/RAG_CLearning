import os           
import chromadb
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
embeddings = SentenceTransformerEmbeddings(model_name="./all-MiniLM-L6-v2", model_kwargs={"device": "cuda"})
chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.create_collection("wiki_freshqa")

path = './data/wiki_docs/'
files = os.listdir(path)
cnt = 1
for file in files:
    with open(path+file,'r') as f:
        raw_text = f.read()
        collection.add(
            documents=[raw_text],
            metadatas=[{"title": file}],
            ids=["id_"+ file ])
        print(cnt, file,len(raw_text.split()))
    cnt += 1