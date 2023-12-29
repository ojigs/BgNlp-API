from flask import Flask, request, jsonify
from transformers import pipeline
import pandas as pd

app = Flask(__name__)
qa_model = pipeline("question-answering", model="deepset/tinyroberta-squad2")

data_frame = pd.read_csv("text_segments.csv")

print(data_frame.head())

@app.route("/")
def helloWorld():
	return "Welcome to BgNLP API Application"

@app.route("/qa", methods=["POST"])
def question_answering():
	data = request.json
	question = data.get("question")
	
	if not question:
		return jsonify({"Error": "Please provide a question"})
	if len(question) > 512:
		return jsonify({"Error": "The question is too long"})

	best_result= None
	best_score = 0.0

	for _, row in data_frame.iterrows():
		context = row["text"]

		try:
			result = qa_model(question=question, context=context)

			if result["score"] > best_score:
				best_result = result["answer"]
				best_score = result["score"]
				
		except ValueError as e:
			print(f"Error processing context: {e}")
			exit(1)

	return jsonify({"answer": best_result, "score": best_score})

if __name__ == '__main__':
	app.run(debug=True)