# BgNlp-API

This project is a Flask-based API application that provides natural language processing (NLP) capabilities for text analysis and question answering. The application uses Elasticsearch as the document store and Haystack for efficient information retrieval.

## Requirements

- Python 3.8+
- Flask
- Flask-Cors
- pandas
- elasticsearch-haystack
- elasticsearch
- transformers (optional, for the QA model)
- OpenAI API key (for the GPTGenerator)

## Installation

1. Install the required packages using pip:

    ```bash
    pip install flask flask-cors pandas elasticsearch-haystack elasticsearch transformers openai
    ```

2. Set the OpenAI API key as an environment variable:

    ```bash
    export OPENAI_API_KEY=your_openai_api_key
    ```

## Usage

1. Download the `text_segments.csv` file and place it in the `data` directory.
2. Run the Flask application:

    ```bash
    python app.py
    ```

The application will start running on [http://localhost:5000](http://localhost:5000).

## API Endpoints

- `GET /`: Welcome message
- `POST /upload`: Uploads documents from the `text_segments.csv` file to the Elasticsearch document store.
- `POST /qa`: Performs question answering using the uploaded documents. The request body should contain a JSON object with a `question` field.
