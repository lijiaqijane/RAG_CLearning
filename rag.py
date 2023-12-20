import os
import torch
from torch import cuda
from time import time
import csv
from utils.pipline import Retrieval, Retrieval_based_QA, Reflection, Execution, Feedback
from utils.kb import get_model, get_embedding, get_vectordb, get_similarity_encoder

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
emb ="./all-MiniLM-L6-v2"
enc = "./bert-base-nli-mean-tokens"
model = "/scratch2/nlp/plm/Llama-2-7b-chat-hf"  # Llama-2-7b-chat-hf  LLaMA-2-7B-32K
llm = get_model(model)
embeddings = get_embedding(emb)
encoder = get_similarity_encoder(enc)
retriever, collection = get_vectordb(embeddings)


filename='./data/freshqa.csv'
max_try = 3 
with open(filename) as csvfile:
    csv_reader = csv.reader(csvfile)  
    header = next(csv_reader)       
    for row in csv_reader:
        ID, q, a, typ = int(row[0]), row[2], row[9], row[4]+'*'+row[6]+'*'+row[7]
        if ID ==0:
            print(ID, q,a)
            no_round = 1
            while True:
                questions, docs = Retrieval(retriever, llm, q)

                ####ToDo: Knowledge_distillation

                prediction = Retrieval_based_QA(retriever, llm, questions, docs)
                
                #feedback = input('feedback: ')
                feedback = Feedback(q, a, prediction, encoder)
                
                if no_round > max_try or feedback =='Answer is CORRECT':
                    pass
                else:
                    reflection = Reflection(q, prediction, feedback)                 
                    Execution(questions, prediction, feedback, reflection, collection)
                    no_round += 1
                    
