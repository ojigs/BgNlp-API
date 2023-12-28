from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)
pipe = pipeline("question-answering", model="deepset/tinyroberta-squad2")

@app.route("/")
def helloWorld():
	return "Welcome to BgNLP API Application"

@app.route("/qa", methods=["POST"])
def question_answering():
	data = request.json
	question = data.get("question")
	context = data.get("context") 
	result = pipe(question=question, context=context)
	return jsonify(result)

if __name__ == '__main__':
	app.run(debug=True)