import os
import torch
from torch import cuda, bfloat16
import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, pipeline
from time import time
import chromadb
from langchain.llms import HuggingFacePipeline
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Chroma
import csv
import time
from sklearn.metrics.pairwise import cosine_similarity

def Retrieval(retriever, llm, question):
    prompt = 'Decompose the question into one or more subquestions without answering. Question: '+ question
    subqs = llm(prompt)
    print('====================', subqs)
    docs = retriever.get_relevant_documents(query=subqs)

    # docs = collection.query(query_texts=q, n_results=2)
    # title, source = docs['metadatas'][0][0]['title'], docs['documents'][0][0]

                
    # docss = vectordb.similarity_search_with_score(q)  #similarity_search_with_score, similarity_search
    # doc_details = docss[0].to_json()['kwargs']
    # title, source = doc_details['metadata']['title'], doc_details['page_content']
    new_question = 'Question: '+ question + subqs.replace('\n',' ')
    return new_question, docs


####ToDo
# def Knowledge_distillation():

def Retrieval_based_QA(retriever, model, question, docs, max_limit=2800):
    try:
        memory = collection.get(ids=["id_memory"])['documents'][0]
    except:
        memory = ''
    print('==================== Memory:'+memory)
    prompt = 'Please answer the below question and explain why. '+ memory + question
    
    ## way1: retrieve, 可能会超长
    # qa = RetrievalQA.from_chain_type(
    # llm=model, 
    # chain_type="map_reduce", 
    # retriever=retriever, 
    # #return_source_documents=True,
    # verbose=True)
    # result = qa.run({"query": prompt})

    
    # way2: given docs directly retrieved as context
    print(len(docs))
    for doc in docs:
        doc_details = doc.to_json()['kwargs']
        print(doc_details['metadata']['title'],len(doc_details['page_content'].split()))


    doc_details = docs[0].to_json()['kwargs']
    text = doc_details['page_content']
    if len(text.split()) > max_limit:
            docs[0].page_content = text[:max_limit]
    chain = load_qa_chain(model, chain_type="stuff")
    result = chain.run(input_documents=[docs[0]], question=prompt)

    # way3: distillated KN
    # chain = load_qa_chain(model, chain_type="stuff")
    # result = chain.run(input_documents=kn, question=prompt)

    return result


def Feedback(question, answer, pred, encoder, sim_rate = 0.9):

    ###way1: GPT eval
    # prompt= [{"role": "system", "content": "Given one question, there is a groundtruth and a predict_answer. \
    #         Please decide whether they are the same or not in semantic. Please output \"Yes" or "No" only."},
    #        {"role": "user", "content": "Question: "+question+'\n'+"groudtruth = " + answer + '\n'+ "predict_answer = "+P}]

    # response = openai.ChatCompletion.create(
    #     model="gpt-4",
    #     messages= p,
    #     temperature = 1.0,
    #     top_p = 1,
    #     max_tokens = 100
    #     )
    # rsp = response['choices'][0]['message']['content']

    ###way2: automatic metrics
    embeddings = encoder.encode([answer, pred])
    similarity = cosine_similarity(embeddings)[0][1]
    print('====================similarity: ', similarity)
    if similarity > sim_rate:
        return 'Answer is CORRECT'
    else:
        return 'Answer is INCORRECT'


def Reflection(llm, question, pred, fd):  
    #shots = ""
    prompt = 'Given the question, there is a model prediction and corresponding feedback. You have 4 choice to deal with the feedback:\
    1.add missing fact to memory, \
    2.delete the false fact in memory, \
    3.add the irrelevant fact to memory, \
    4.do nothing if facts OK but bad reasoning, \
    5.add the clarification of the question to memory. \
    Please determine and output the number 1/2/3/4/5. '
    
    case = "Q: "+ question+' Prediction: '+pred+ ' Feedback: '+fd + ' '+ prompt
    result = llm(prompt+case)
    return result


def Execution(question, pred, fd, rlt, collection):
    try:
        memory = collection.get(ids=["id_memory"])['documents'][0]
        insert_text = memory + ' ' + pred + ' ' + fd + ' ' + rlt
    except:
        memory = ''
        insert_text = memory + ' ' + question + ' ' + pred + ' ' + fd + ' ' + rlt
    collection.upsert(
            documents=[insert_text],
            metadatas=[{"title": 'memory'}],
            ids=["id_memory"])
