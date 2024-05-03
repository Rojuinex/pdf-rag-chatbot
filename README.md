# PDF RAG Chatbot

A simple RAG chatbot to answer questions about your documents.

## Dependencies

* Python 12
* Ollama or an OpenAI API key

## Installation

1. Create a virtual environment `python -m venv .venv`
2. Activate environment `source .venv/bin/activate`
3. Install dependencies `pdm install`

## Usage

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

By default the application will assume Ollama and llama3 are installed.  You can do that by:

1. Download [Ollama here](https://ollama.com/).
2. Start the server with llama3 `ollama run llama3`

If you have an OpenAI API key and want to use that instead you can run the application by
setting the `OPENAI_API_KEY` environment variable specifying any OpenAI chat model that
begins with `gpt`.  However, it should be noted that the prompts were developed with llama3
and may need to be tweaked to get good performance when specifying a different model.

```shell
$  export OPENAI_API_KEY=`...`
$ pdf-rag-chatbot --model gpt-3.5-turbo
```