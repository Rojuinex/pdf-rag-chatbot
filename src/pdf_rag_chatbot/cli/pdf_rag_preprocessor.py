import os
import sys
import asyncio
from loguru import logger

import click

logger.remove()
logger.add(sys.stderr, level=os.environ.get("LOGURU_LEVEL", "INFO"))

@click.command(context_settings={'show_default': True})
@click.option("--db", "db_path", default="warehouse.duckdb", help="Path to the duckdb database file.")
@click.argument("file_path", type=click.Path(exists=True))
def main(db_path: str, file_path: str):
    import duckdb
    from pdf_rag_chatbot.db import setup_database
    from pdf_rag_chatbot.data_pipeline.text_pipeline import TextPipeline
    from pdf_rag_chatbot.data_pipeline.messages import FileUploaded


    db = duckdb.connect(db_path)
    setup_database(db)

    pipeline = TextPipeline(db)


    # If we're given a single file, we can process it directly.
    if not os.path.isdir(file_path):
        pipeline(FileUploaded(file_path=file_path))
    else:
        for root, dirs, files in os.walk(file_path):
            for file in files:
                if file.split(".")[-1] not in ["pdf", "text", "txt"]:
                    continue

                file_path = os.path.join(root, file)
                pipeline(FileUploaded(file_path=file_path))
