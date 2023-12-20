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


import func_timeout
from func_timeout import func_set_timeout
@func_set_timeout(240)
def test_model(tokenizer, pipeline, prompt_to_test):
    try:
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
    except:
        return None



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


# retriever = WikipediaRetriever()
# filename='./freshqa.csv'
# with open(filename) as csvfile:
#     csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
#     header = next(csv_reader)        # 读取第一行每一列的标题
#     for row in csv_reader:
#         ID, q, a, typ = int(row[0]), row[2], row[9], row[4]+'*'+row[6]+'*'+row[7]
#         if ID >= 0:
#             print(ID, q,a)
#             prediction = RAG(retriever, llm, q)
#             #feedback = GPT_assis(q, a, prediction)
#             print(prediction)
#             print('\n')


filename='./freshqa.csv'
with open(filename) as csvfile:
    csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
    header = next(csv_reader)        # 读取第一行每一列的标题
    for row in csv_reader:
        ID, q, a, typ = int(row[0]), row[2], row[9], row[4]+'*'+row[6]+'*'+row[7]
        if ID >= 205:
            print(ID, q,a)

            try:
                time_1 = time.time()
                prediction = test_model(tokenizer,query_pipeline,q)
                #feedback = GPT_assis(q, a, prediction)
                time_2 = time.time()
                print(f"-------------Time cost: {round(time_2-time_1, 3)} sec.")
            except func_timeout.exceptions.FunctionTimedOut:
                prediction=''

            print(prediction)
            print('\n')
            with open('./result/freshqa_llamKN_result.txt','a+') as g:
                g.write(str(ID)+': '+ q + a+'\n')
                g.write(prediction+'\n')
                g.write('\n')


















































































































                