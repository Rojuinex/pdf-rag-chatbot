# PDF RAG Chatbot

A simple RAG chatbot to answer questions about your documents.

## Dependencies

* Python 12
* Ollama

## Installation

1. Create a virtual environment `python -m venv .venv`
2. Activate environment `source .venv/bin/activate`.
3. Install dependencies `pdm install`.
4. Download spaCy model `python -m spacy download en_core_web_trf`.

## Usage

First ensure the Ollama server is running

1. Download [Ollama](https://ollama.com/).
2. Start the server `ollama run llama3`

Launch the server using:

```shell
$ pdf-rag-chatbot --help
Usage: pdf-rag-chatbot [OPTIONS]

Options:
  --port INTEGER  Port to run the server on.
  --db TEXT       Path to the duckdb database file.
  --model TEXT    The language model to use for agents.
  --help          Show this message and exit.
```


## Common issues

### spaCy complains about not being able to find the pip package in the virtual environment

```sh
$ python -m ensurepip
```
