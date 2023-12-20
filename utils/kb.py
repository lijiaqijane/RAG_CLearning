import os
import torch
from torch import cuda, bfloat16
import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
from time import time
import chromadb
from langchain.llms import HuggingFacePipeline
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Chroma
import csv
import time

def get_model(model): 
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
    )

    model_config = AutoConfig.from_pretrained(model)
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForCausalLM.from_pretrained(
        model,
        trust_remote_code=True,
        config=model_config,
        #load_in_8bit=True
        quantization_config=bnb_config,
    )#.half().to(device)
    query_pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            repetition_penalty=1.3,
            max_new_tokens = 300,
            # torch_dtype=torch.float16,
            device_map="auto")
    llm = HuggingFacePipeline(pipeline=query_pipeline)
    return llm

def get_embedding(model_name):
    embeddings = SentenceTransformerEmbeddings(model_name = model_name)
    return embeddings
    
def get_vectordb(embeddings, re_no=4, collection_name="wiki_freshqa"):
    chroma = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma.get_collection(collection_name)
    vectordb = Chroma(
        client=chroma,
        collection_name= collection_name,
        embedding_function=embeddings,
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": re_no})
    return retriever, collection

def get_similarity_encoder(encode_model):
    encoder = SentenceTransformer(encode_model)
    return encoder