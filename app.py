import os
from getpass import getpass
from flask import Flask, request, jsonify
from flask_cors import CORS
# from transformers import pipeline
import pandas as pd
from elasticsearch_haystack.document_store import ElasticsearchDocumentStore
from elasticsearch_haystack.embedding_retriever import ElasticsearchEmbeddingRetriever
from haystack import Document, Pipeline
from haystack.components.generators import GPTGenerator

app = Flask(__name__)
CORS(app)
# qa_model = pipeline("question-answering", model="deepset/tinyroberta-squad2")

data_frame = pd.read_csv("data/text_segments.csv")

print(data_frame.head())




@app.route("/")
def hello_world():
    return "Welcome to BgNLP API Application"


@app.route("/upload")
def upload_document():
    document = [Document(content=text, content_type="table", meta={"pagenum": pagenum, "doc_name": doc_name})
                for text, pagenum, doc_name in zip(data_frame["text"], data_frame["pagenum"], data_frame["doc_name"])]
    document_store = ElasticsearchDocumentStore(hosts="http://localhost:9200")
    document_store.write_documents(document)
    print(document_store.count_documents())


@app.route("/qa", methods=["POST"])
def question_answering():
    question = request.get_json("question")
    document_store = ElasticsearchDocumentStore(hosts="http://localhost:9200")
    retriever = ElasticsearchEmbeddingRetriever(document_store=document_store)
    openai_api_key = os.getenv("OPENAI_API_KEY", None) or getpass("Enter OpenAI API Key")
    generator = GPTGenerator(api_key=openai_api_key)

    query_pipeline = Pipeline()
    query_pipeline.add_component("retriever", retriever)
    query_pipeline.add_component("llm", generator)
    query_pipeline.connect("retriever", "llm")

    query_pipeline.draw("query_pipeline.png")

    prediction = query_pipeline.run({"retriever": {"query": question}})

    response = jsonify(prediction)

    return response


# @app.route("/qa", methods=["POST"])
# def question_answering():
#     data = request.json
#     question = data.get("question")
#
#     if not question:
#         return jsonify({"Error": "Please provide a question"})
#     if len(question) > 512:
#         return jsonify({"Error": "The question is too long"})
#
#     best_result = None
#     best_score = 0.0
#
#     for _, row in data_frame.iterrows():
#         context = row["text"]
#
#         try:
#             result = qa_model(question=question, context=context)
#
#             if result["score"] > best_score:
#                 best_result = result["answer"]
#                 best_score = result["score"]
#
#         except ValueError as e:
#             print(f"Error processing context: {e}")
#             exit(1)
#
#     return jsonify({"answer": best_result, "score": best_score})


if __name__ == '__main__':
    app.run(debug=True)
