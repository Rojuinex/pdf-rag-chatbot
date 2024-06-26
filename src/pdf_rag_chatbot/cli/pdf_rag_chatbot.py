import os
import sys
from loguru import logger

import click


logger.remove()
logger.add(sys.stderr, level=os.environ.get("LOGURU_LEVEL", "INFO"))

@click.command(context_settings={'show_default': True})
@click.option("--port", default=5000, help="Port to run the server on.")
@click.option("--db", default="warehouse.duckdb", help="Path to the duckdb database file.")
@click.option("--model", default="llama3", help="The language model to use for agents.")
def main(port: int, db: str, model: str):
    from pdf_rag_chatbot.app import App
    import polars as pl

    pl.Config(
        fmt_str_lengths = 300,
        tbl_width_chars = 500
    )

    if model.startswith("gpt"):
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=model)
    else:
        from langchain_community.chat_models import ChatOllama
        llm = ChatOllama(model=model)

    app = App(
        database=db,
        llm=llm,
    )
    app.launch(
        server_port=port,
    )


if __name__ == "__main__":
    main()
