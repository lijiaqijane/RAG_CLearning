from torch import cuda, bfloat16
import torch
import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, pipeline
from time import time
import chromadb
from chromadb.config import Settings
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.vectorstores import Chroma, FAISS
from langchain.retrievers import WikipediaRetriever
from langchain.embeddings import OpenAIEmbeddings
import csv
import time
import jieba.analyse
import os
import openai
os.environ["OPENAI_API_KEY"] = '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'
openai.api_key = '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'

import chromadb
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection_name = "wiki_Dec_2023"
collection = chroma_client.get_collection(collection_name)
# #chroma_client = chromadb.Client()
#collection = chroma_client.create_collection(name=collection_name)

## quey current DB
print(collection.count())
# res = collection.get(
# include=["metadatas"]
# # ids=["id86"]
# )
#print(res)
# filename='./freshqa.csv'
# with open(filename) as csvfile:
#     csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
#     header = next(csv_reader)        # 读取第一行每一列的标题
#     for row in csv_reader:
#         ID, q, a, typ = int(row[0]), row[2], row[9], row[4]+'*'+row[6]+'*'+row[7]
#         if ID >= 0:
#             print(ID, q,a)
#             query_results = collection.query(
#             query_texts=[q],
#             n_results=3,
#             )
#             for i in list(query_results.keys()):
#                 print(query_results[i])

## delete DB
# collection.delete(
#     ids=["id0"]
# )


### insert DB
# raw_text = ''
# with open('./freshqa_groundtruth.txt','r') as f:
#     raw_text += i for i in f.readlines()

# collection.add(
# documents=["Elon Musk has not been X Corp.'s CEO for any length of time as he is not currently the CEO of the company."],
# metadatas=[{"source": "wiki"}],
# ids=["id0"])
