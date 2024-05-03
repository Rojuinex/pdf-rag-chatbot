import asyncio
import traceback
from typing import Union, Any, List, Union, Type

from duckdb import DuckDBPyConnection

from pdf_rag_chatbot.data_pipeline.messages import (
    DeadLetterMessage,
    Message,
)

class PipelineStep:
    def __init__(
        self,
        name: str,
        request_type: Union[Type[Message] | List[Type[Message]]],
        db: DuckDBPyConnection
    ):
        self.name = name
        self.request_type = request_type
        self.db = db

    def __call__(self, request: Any) -> Union[Message, List[Message], None]:
        """Process a request and return the result."""
        raise NotImplementedError

    def _raise_for_request_type(self, request: Any):
        if isinstance(self.request_type, list):
            if not any(isinstance(request, t) for t in self.request_type):
                raise ValueError("Invalid request type.")
        elif not isinstance(request, self.request_type):
            raise ValueError("Invalid request type.")

    async def run(
        self,
        input_queue: asyncio.Queue,
        output_queue: asyncio.Queue,
        deadletter_queue: asyncio.Queue,
        shutdown_event: asyncio.Event,
    ):
        while not shutdown_event.is_set():
            try:
                req = await input_queue.get()

                self._raise_for_request_type(req)

                result = self(req)

                if result is None:
                    continue
                elif not isinstance(result, list):
                    result = [result]

                for r in result:
                    await output_queue.put(r)

            except asyncio.CancelledError:
                break
            except Exception as e:
                await deadletter_queue.put(
                    DeadLetterMessage(
                        step=self.name,
                        error=str(e),
                        traceback=traceback.format_exc(),
                        request=req,
                    )
                )