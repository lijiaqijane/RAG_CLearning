import os
import openai
os.environ["OPENAI_API_KEY"] = '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'
openai.api_key = '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'
os.environ["SERPAPI_API_KEY"] = '4c962f7a7841fe021ce38682a3f00dfa460d40df6345f268b72980ca0a1db65c'
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

def Retrieval(retriever, llm, Q):
    prompt = Q+' Decompose the question into several sub-questions step by step.'
    subqs = llm(prompt)
    docs = retriever.get_relevant_documents(query=subqs)

    # docs = collection.query(query_texts=q, n_results=2)
    # title, source = docs['metadatas'][0][0]['title'], docs['documents'][0][0]
    # print(title)
    # print(source[:100])
                
    # docss = vectordb.similarity_search_with_score(q)  #similarity_search_with_score, similarity_search
    # doc_details = docss[0].to_json()['kwargs']
    # title, source = doc_details['metadata']['title'], doc_details['page_content']
    return subqs, docs


def Retrieval_based_QA(retriever, model, Q, docs, max_limit=2800):
    try:
        context = collection.get(ids=["id_context"])['documents'][0]
    except:
        context = ''
    print('Context:'+context)
    prompt = 'Please answer the below question and explain why. '+ context + Q
    
    ## way1: retrieve, 可能会超长
    # qa = RetrievalQA.from_chain_type(
    # llm=model, 
    # chain_type="map_reduce", 
    # retriever=retriever, 
    # #return_source_documents=True,
    # verbose=True)
    # result = qa.run({"query": prompt})

    
    # way2: given docs directly retrieved as context
    doc_details = docs[0].to_json()['kwargs']
    text = doc_details['page_content']
    if len(text.split()) > max_limit:
            docs[0].page_content = text[:max_limit]
    chain = load_qa_chain(model, chain_type="stuff")
    result = chain.run(input_documents=docs, question=prompt)

    # way3: composed kn
    # chain = load_qa_chain(model, chain_type="stuff")
    # result = chain.run(input_documents=kn, question=prompt)

    return result

def Reflection(llm, question, pred, kn):  
    #shots = ""
    prompt = 'Given the question, there is a prediction and corresponding human feedback. You have 4 choice to deal with the feedback:\
    1.add missing fact to memory, \
    2.delete the false fact in memory, \
    3.add the irrelevant fact to memory, \
    4.do nothing if facts OK but bad reasoning, \
    5.add the clarification of the question to memory. \
    Please determine and output the number 1/2/3/4/5. '
    
    shots = 'Q: Can a magnet attract a penny? Pred: Yes, because a magnet can attract magnetic metals and a penny is made of magnetic metals. Feedback: A penny is made of copper. Action: 1. add missing fact to memory \
    Q: Can a magnet attract a penny? Pred: Yes, because a penny is made of copper and a magnet can attract copper. Feedback: A magnet can\'t attract copper. Output: 2 \
    Q: Can a magnet attract a penny? Pred: Yes, a magnet can attract copper and metals are shiny and lustrous. Feedback: Metals are shiny and lustrous is irelevant to the question. Output: 3 \
    Q: Can a magnet attract a penny? Pred: Yes, a magnet can attract magnetic metals and a penny is made of copper. Feedback: Facts OK but bad reasoning. Output: 4 \
    Q: Can a magnet attract a penny? Pred: The penny are sometimes shiny and lustrous. Feedback: The question is asking about the relation between magnet and penny. Output: 5 '
    
    case = "Q: "+ question+' Prediction: '+pred+ ' Feedback: '+kn + ' '+ shots
    result = llm(prompt+case)
    return result

def Execution(act, question, kn, file, original):
    if act=='1':
        collection.upsert(
            documents=[kn+' '+original],
            metadatas=[{"title": file}],
            ids=["id_"+ file ])
        tmp = collection.get(
                        ids=["id_"+ file]
                    )['documents'][0]
        print(tmp[:200])
    elif act=='2':
        #collection.update(
        pass
    elif act in ['3','5']:
        try:
            context = collection.get(ids=["id_context"])['documents'][0]
        except:
            context = ''
        
        collection.upsert(
            documents=[context+' Given question: '+question+kn],
            metadatas=[{"title": 'context'}],
            ids=["id_context"])

    elif act=='4':
        pass


model = "/scratch2/nlp/plm/Llama-2-7b-chat-hf" # Llama-2-7b-chat-hf  LLaMA-2-7B-32K
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

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
        repetition_penalty=1.1,
        max_new_tokens = 300,
        # torch_dtype=torch.float16,
        device_map="auto")
llm = HuggingFacePipeline(pipeline=query_pipeline)

embeddings = SentenceTransformerEmbeddings(model_name="./all-MiniLM-L6-v2")
chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_collection("wiki_freshqa")
vectordb = Chroma(
    client=chroma,
    collection_name="wiki_freshqa",
    embedding_function=embeddings,
)
retriever = vectordb.as_retriever(search_kwargs={"k": 4})

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
                subquestions, docs = Retrieval(retriever,llm, q)
                print(f"=====Retrieved documents: "+str(len(docs)))
                print('------------', subquestions)

                #external_kn = Summarization(docs)

                time_1 = time.time()
                prediction = Retrieval_based_QA(retriever, llm, subquestions, docs)
                time_2 = time.time()
                print('------------Prediction '+ prediction)
                print('-----Round '+str(no_round)+', Time cost '+str(round(time_2-time_1, 2))+'sec.')

                
                human_feedback = input('human_feedback: ')
                print('------------Feedback '+human_feedback)
                
                
                if no_round > max_try or human_feedback =='Success':
                    collection.delete(ids=["id_context"])
                    break
                else:
                    time_3 = time.time()
                    action = Reflection(llm, q, prediction, human_feedback)
                    print('------------Action '+action)
                    time_4 = time.time()
                    action = input('action: ')
                    print('------------Action '+action+', Time cost '+str(round(time_4-time_3, 2))+'sec.')
                    
                    Execution(action, q, human_feedback, title, source)
                    no_round += 1
                    
