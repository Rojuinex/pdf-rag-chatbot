import asyncio
from typing import Callable

from duckdb import DuckDBPyConnection

from pdf_rag_chatbot.data_pipeline.messages import (
    DeadLetterMessage,
    FileUploaded,
)
from pdf_rag_chatbot.data_pipeline.steps import (
    Ingest,
    NLP,
    Embed,
)

DeadLetterHandler = Callable[[DeadLetterMessage], None]

class TextPipeline:
    def __init__(
        self,
        db: DuckDBPyConnection,
    ):
        self.db = db

        self.ingest = Ingest(db)
        self.nlp = NLP(db)
        self.embed = Embed(db)


    def __call__(self, req: FileUploaded):
        """Process a file and return the extracted information.

        Args:
            file_path (str): The name of the file to process.

        Returns:
            dict: The extracted information.
        """

        res = self.ingest(req)
        if res is None:
            return
        
        res = self.nlp(res)

        for r in res:
            self.embed(r)

    async def start(self):
        """Start the pipeline."""
        self.input_queue = asyncio.Queue()
        self.deadletter_queue = asyncio.Queue()
        self.shutdown_event = asyncio.Event()

        return asyncio.create_task(
            self.run(self.input_queue, self.deadletter_queue, self.shutdown_event)
        )

    async def stop(self):
        """Stop the pipeline."""
        self.shutdown_event.set()

    async def put(self, req: FileUploaded):
        """Put a request into the pipeline."""
        await self.input_queue.put(req)

    async def add_deadletter_handler(self, handler: DeadLetterHandler):
        """Add a deadletter handler to the pipeline."""
        async def _handler():
            while not self.shutdown_event.is_set():
                msg = await self.deadletter_queue.get()
                await handler(msg)

        return asyncio.create_task(_handler())

    async def run(
        self,
        input_queue: asyncio.Queue,
        deadletter_queue: asyncio.Queue,
        shutdown_event: asyncio.Event,
    ):
        ingest_result_queue = asyncio.Queue()
        nlp_result_queue = asyncio.Queue()

        ingest_task = asyncio.create_task(
            self.ingest.run(input_queue, ingest_result_queue, deadletter_queue, shutdown_event)
        )

        nlp_task = asyncio.create_task(
            self.nlp.run(ingest_result_queue, nlp_result_queue, deadletter_queue, shutdown_event)
        )

        embed_task = asyncio.create_task(
            self.embed.run(nlp_result_queue, input_queue, deadletter_queue, shutdown_event)
        )

        await asyncio.gather(ingest_task, nlp_task, embed_task)
