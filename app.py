import os
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
import openai
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.document_loaders.json_loader import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA



load_dotenv()

app = Flask(__name__)

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

directory = '/home/hstpl_lap_184/Music/chroma-langchain-demo/data/faqs.json'

loader = JSONLoader(directory, jq_schema='.content', text_content=False)
documents = loader.load()

# Split documents
text_splitter = CharacterTextSplitter(
    separator="}",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)
docs = text_splitter.split_documents(documents)

# Embeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Vector store (Chroma)
db = Chroma.from_documents(docs, embeddings)

# OpenAI API Key
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# Chat model
model_name = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name=model_name)

# QA Chain
chain = load_qa_chain(llm, chain_type="stuff", verbose=True)

# Retrieval QA Chain
retrieval_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=db.as_retriever())

# API endpoints

@app.route('/question_answering', methods=['POST'])
def question_answering():
    query = request.json['query']
    matching_docs = db.similarity_search(query)
    answer = chain.run(input_documents=matching_docs, question=query)
    final_answer = retrieval_chain.run(query)
    return jsonify(final_answer)



if __name__ == '__main__':
    app.run(debug=True)



