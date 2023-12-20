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


def RAG(retriever, model, Q):
    qa = RetrievalQA.from_chain_type(
    llm=model, 
    chain_type="stuff", 
    retriever=retriever, 
    verbose=True)
    query = Q
    result = qa.run(query)
    return result

def GPT_assis(Q, A, P):
    prompt1= [{"role": "system", "content": "Given one question, there is a groundtruth and a predict_answer. Please decide whether they are the same or not in semantic. Please output True or False only."},
           {"role": "user", "content": "Question: "+Q+'\n'+"groudtruth = " + A + '\n'+ "predict_answer = "+P}]
    
    prompt2= [
    {"role": "system","content":"Please generatge a groundtruth based on given question and answer."},
    {"role": "user","content":"Input: question="+Q},
    {"role": "user","content":"Input: answer="+A}]
    
    feedback = set()
    for p in [prompt1, prompt2]:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages= p,
            temperature = 1.0,
            top_p = 1,
            max_tokens = 2048
            )
        rsp = response['choices'][0]['message']['content']
        feedback.add(rsp)
    return feedback

def test_model(tokenizer, pipeline, prompt_to_test):

    sequences = pipeline(
        prompt_to_test,
        do_sample=True,
        temperature = 1.0,
        top_p = 1,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=500)

    response = sequences[0]['generated_text']
    return response



model = "/scratch2/nlp/plm/Llama-2-7b-chat-hf"
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
model_config = AutoConfig.from_pretrained(model)
tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForCausalLM.from_pretrained(
    model,
    trust_remote_code=True,
    config=model_config
)
query_pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length = 300,
        torch_dtype=torch.float16,
        device_map="auto",)
llm = HuggingFacePipeline(pipeline=query_pipeline)

model_name = "./all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": "cuda"})
vectordb = Chroma(collection_name = "wiki_Dec_2023", persist_directory="./chroma_db", embedding_function=embeddings)
#print(vectordb.get_collection("wiki_Dec_2023").count())
retriever = vectordb.as_retriever()


filename='./freshqa.csv'
with open(filename) as csvfile:
    csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
    header = next(csv_reader)        # 读取第一行每一列的标题
    for row in csv_reader:
        ID, q, a, typ = int(row[0]), row[2], row[9], row[4]+'*'+row[6]+'*'+row[7]
        if  51> ID >= 50:
            print(ID, q,a)
            #time_1 = time.time()
            prediction = RAG(retriever, llm, q)
            #feedback = GPT_assis(q, a, prediction)
            #time_2 = time.time()
            # print(f"---------------Time cost: {round(time_2-time_1, 3)} sec.")
            # print(prediction)
            # print('\n')
            # with open('./result/freshqa_llamRAG_result.txt','a+') as g:
            #     g.write(str(ID)+': '+ q + a+'\n')
            #     g.write(prediction+'\n')
            #     g.write('\n')

            docs = vectordb.similarity_search(q)
            print(f"Retrieved documents: {len(docs)}")
            for doc in docs:
                doc_details = doc.to_json()['kwargs']['page_content']
                print("Text: ", len(doc_details), doc_details[:500], "\n")
