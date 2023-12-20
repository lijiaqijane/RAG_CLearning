from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
import os
import chromadb

# ##Refer: https://juejin.cn/post/7249942474591699004

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection_name = "wiki_Dec_2023"
collection = chroma_client.get_collection(collection_name)

####1: 单个wiki_all直接TextLoader load
# loader = TextLoader("../wiki.txt",encoding="utf8")
# documents = loader.load()
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
# splits = text_splitter.split_documents(documents)
# print(len(splits))
# model_name = "./all-mpnet-base-v2"
# embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": "cuda"})
# vectordb = Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory="chroma_db")
# retriever = vectordb.as_retriever()


####2: 生成kiki_all的slices文件夹
# f = open('./wiki_all.txt', 'r') 
# raw_text = ''
# cnt = 2400001
# for line in f.readlines():
#     line = eval(line)
#     raw_text += 'Title: '+line['title']+' Text: '+line['text']
#     if cnt % 100000 ==0:
#         print(cnt)
#         with open('./wiki_slices/'+str(cnt)+'.txt', 'a+')  as g:
#             g.write(raw_text)
#         raw_text = ''
#     cnt += 1

####2-1: 基于每个slice生成DB,并合并
# text_splitter = RecursiveCharacterTextSplitter(chunk_overlap=20)
# model_name = "./all-mpnet-base-v2"
# embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": "cuda"})
# def split_list(input_list, chunk_size):
#     for i in range(0, len(input_list), chunk_size):
#         yield input_list[i:i + chunk_size]
    
# db_num = 1
# path = './wiki_slices/'
# files = os.listdir(path)
# for i in files:
#     print(i)
#     loader = TextLoader(path+i,encoding="utf8")
#     documents = loader.load()
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
#     splits = text_splitter.split_documents(documents)
#     split_docs_chunked = split_list(splits, 41000)
#     print(len(splits))

#     if db_num == 1:
#         for split_docs_chunk in split_docs_chunked:
#             print(len(split_docs_chunk))
#             db =  Chroma.from_documents(documents=split_docs_chunk, embedding=embeddings, persist_directory="chroma_db")
#     else:
#         for split_docs_chunk in split_docs_chunked:
#             print(len(split_docs_chunk))
#             db1 = Chroma.from_documents(documents=split_docs_chunk, embedding=embeddings, persist_directory="chroma_db")


#         db1_data=db1._collection.get(include=['documents','metadatas','embeddings'])
#         db._collection.add(
#             embeddings=db1_data['embeddings'],
#             metadatas=db1_data['metadatas'],
#             documents=db1_data['documents'],
#             ids=db1_data['ids']
#         )
#     db_num+=1



# ####2-2: 基于每个slice DB.add
# chroma_client = chromadb.PersistentClient(path="./chroma_db")
# collection_name = "wiki_Dec_2023"
# #chroma_client = chromadb.Client()
# #collection = chroma_client.create_collection(name=collection_name)
# collection = chroma_client.get_collection(collection_name)

# white_list = ['id25','id1','id52', 'id15', 'id42', 'id8', 'id7', 'id72', 'id35', 'id84', 'id62', 'id12', 'id45', 'id55', 'id32', 'id83', 'id65', 'id22', 'id75', 'id24', 'id73', 'id6', 'id9', 'id34', 'id63', 'id85', 'id53', 'id14', 'id43', 'id33', 'id64', 'id82', 'id23', 'id74', 'id13', 'id44', 'id54', 'id47', 'id48', 'id10', 'id58', 'id57', 'id68', 'id67', 'id81', 'id30', 'id77', 'id78', 'id2', 'id20', 'id50', 'id40', 'id18', 'id17', 'id70', 'id5', 'id28', 'id27', 'id60', 'id86']
# cnt = 1
# path = './wiki_slices/'
# files = os.listdir(path)
# for i in files:
#     f = open(path+i, 'r') 
#     raw_text = str(f.read())
#     ID = str(int(i.strip('.txt'))//100000)
#     print('id'+ID)
#     if 'id'+ID not in white_list:
#         print(i)
#         try:
#             collection.add(
#                 documents=[raw_text],
#                 metadatas=[{"source": "wiki"}],
#                 ids=["id"+ID ]
#             )
#         except:
#             print("!!!!!!!!!!!!!!!!!!!"+ID)
#             pass
#         print(cnt, ID)
#         white_list.append('id'+ID)
#         cnt += 1

###3: 每个doc一个ID DB.add
f = open('./data/wiki_all.txt', 'r') 
cnt = 1
for line in f.readlines():
    line = eval(line)
    if cnt > 781367:
        raw_text = 'Title: '+line['title']+' Text: '+line['text']
        collection.add(
            documents=[raw_text],
            metadatas=[{"title": line['title']}],
            ids=["id"+str(cnt) ])
        print(cnt, line['title'] )
    cnt += 1
