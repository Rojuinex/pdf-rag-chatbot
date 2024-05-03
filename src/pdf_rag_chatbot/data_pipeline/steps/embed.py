import hashlib
import torch

from duckdb import DuckDBPyConnection
from sentence_transformers import SentenceTransformer

from pdf_rag_chatbot.data_pipeline.steps.pipeline_step import PipelineStep
from pdf_rag_chatbot.data_pipeline.messages import (
    SentenceCreated,
    EntityCreated,
)

class Embed(PipelineStep):
    def __init__(
        self,
        db: DuckDBPyConnection,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        super().__init__(
            "embed",
            request_type=[SentenceCreated, EntityCreated],
            db=db
        )

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device("mps")

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=self.device)

    def __call__(self, req: SentenceCreated | EntityCreated) -> None:
        if isinstance(req, SentenceCreated):
            text = req.sentence.text
        else:
            text = req.entity.text

        cased_text_hash = hashlib.md5(text.encode()).hexdigest()

        r = self.db.execute(
            """--sql
                SELECT
                    COUNT(*) > 0 AS exists
                FROM text_embedding
                WHERE
                    cased_text_hash = ?
                    AND model_name = ?
            """,
            (cased_text_hash, self.model_name)
        ).fetchone()

        if r[0] == True:
            return

        uncased_text_hash = hashlib.md5(text.lower().encode()).hexdigest()

        embedding = self.model.encode([text])[0].tolist()

        self.db.execute(
            """--sql
                INSERT INTO text_embedding (
                    cased_text_hash,
                    uncased_text_hash,
                    model_name,
                    text,
                    embedding
                )
                VALUES (?, ?, ?, ?, ?)
            """,
            (cased_text_hash, uncased_text_hash, self.model_name, text, embedding)
        )

        return

