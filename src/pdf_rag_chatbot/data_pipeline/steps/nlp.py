from datetime import datetime
import hashlib
from typing import List, Optional, Dict

from duckdb import DuckDBPyConnection
import spacy

from pdf_rag_chatbot.data_pipeline.steps.pipeline_step import PipelineStep
from pdf_rag_chatbot.data_pipeline.messages import (
    DocumentCreated,
    SentenceCreated,
    EntityCreated,
)
from pdf_rag_chatbot.db.models import (
    DocumentEntity,
    DocumentSentence,
    Entity,
    Sentence,
)

class NLP(PipelineStep):
    def __init__(
        self,
        db: DuckDBPyConnection,
        spacy_model: str = "en_core_web_trf",
    ):
        super().__init__("nlp", request_type=DocumentCreated, db=db)
        self.nlp = spacy.load(spacy_model)

    def __call__(self, req: DocumentCreated) -> Optional[List[SentenceCreated | EntityCreated]]:
        document_hash = req.document.document_hash
        doc = self.nlp(req.document.text)

        document_entities: List[DocumentEntity] = []
        document_sentences: List[DocumentSentence] = []

        for sent_idx, sent in enumerate(doc.sents):
            ds = DocumentSentence(
                document_hash=document_hash,
                cased_sentence_hash=hashlib.md5(sent.text.encode()).hexdigest(),
                uncased_sentence_hash=hashlib.md5(sent.text.lower().encode()).hexdigest(),
                text=sent.text,
                index=sent_idx,
                start_char=sent.start_char,
                end_char=sent.end_char,
                processed_at=datetime.now(),
            )

            document_sentences.append(ds)

            for ent in sent.ents:
                de = DocumentEntity(
                    document_hash=document_hash,
                    cased_entity_hash=hashlib.md5(ent.text.encode()).hexdigest(),
                    uncased_entity_hash=hashlib.md5(ent.text.lower().encode()).hexdigest(),
                    text=ent.text,
                    sentence_index=sent_idx,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                    label=ent.label_,
                    processed_at=datetime.now(),
                )

                document_entities.append(de)


        self.db.executemany(
            """--sql
            INSERT INTO document_sentence (
                document_hash,
                cased_sentence_hash,
                uncased_sentence_hash,
                text,
                index,
                start_char,
                end_char,
                processed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    d.document_hash,
                    d.cased_sentence_hash,
                    d.uncased_sentence_hash,
                    d.text,
                    d.index,
                    d.start_char,
                    d.end_char,
                    d.processed_at,
                )
                for d in document_sentences
            ],
        )

        self.db.executemany(
            """--sql
            INSERT INTO document_entity (
                document_hash,
                cased_entity_hash,
                uncased_entity_hash,
                text,
                sentence_index,
                start_char,
                end_char,
                label,
                processed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    d.document_hash,
                    d.cased_entity_hash,
                    d.uncased_entity_hash,
                    d.text,
                    d.sentence_index,
                    d.start_char,
                    d.end_char,
                    d.label,
                    d.processed_at,
                )
                for d in document_entities
            ],
        )


        # Get all of the new sentences and entities that were added
        # to the database that don't exist in the sentence or entity tables
        new_objs = self.db.execute(
            """--sql
                SELECT
                    'sentence' AS obj_type,
                    cased_sentence_hash AS text_hash
                FROM document_sentence
                WHERE document_hash = ?
                AND cased_sentence_hash NOT IN (
                    SELECT cased_sentence_hash FROM sentence
                )
                UNION ALL
                SELECT
                    'entity' AS obj_type,
                    cased_entity_hash AS text_hash
                FROM document_entity
                WHERE document_hash = ?
                AND cased_entity_hash NOT IN (
                    SELECT cased_entity_hash FROM entity
                )
            """,
            (document_hash, document_hash),
        ).fetchall()

        obj_map = {
            "sentence": {},
            "entity": {},
        }

        for obj_type, text_hash in new_objs:
            obj_map[obj_type][text_hash] = True

        out_messages : List[SentenceCreated | EntityCreated] = []
        added_sentences : Dict[str, Sentence] = {}
        added_entities : Dict[str, Entity] =  {}

        for ds in document_sentences:
            if ds.cased_sentence_hash in obj_map["sentence"]:
                if ds.cased_sentence_hash not in added_sentences:
                    sent = Sentence(
                        cased_sentence_hash=ds.cased_sentence_hash,
                        uncased_sentence_hash=ds.uncased_sentence_hash,
                        text=ds.text,
                        processed_at=ds.processed_at,
                    )
                    added_sentences[ds.cased_sentence_hash] = sent
                    out_messages.append(SentenceCreated(sentence=sent))

        for de in document_entities:
            if de.cased_entity_hash in obj_map["entity"]:
                if de.cased_entity_hash not in added_entities:
                    ent = Entity(
                        cased_entity_hash=de.cased_entity_hash,
                        uncased_entity_hash=de.uncased_entity_hash,
                        text=de.text,
                        label=de.label,
                        processed_at=de.processed_at,
                    )
                    added_entities[de.cased_entity_hash] = ent
                    out_messages.append(EntityCreated(entity=ent))

        self.db.executemany(
            """--sql
            INSERT INTO sentence (
                cased_sentence_hash,
                uncased_sentence_hash,
                text,
                processed_at
            )
            VALUES (?, ?, ?, ?)
            """,
            [
                (
                    s.cased_sentence_hash,
                    s.uncased_sentence_hash,
                    s.text,
                    s.processed_at,
                )
                for s in added_sentences.values()
            ],
        )

        self.db.executemany(
            """--sql
            INSERT INTO entity (
                cased_entity_hash,
                uncased_entity_hash,
                text,
                label,
                processed_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    e.cased_entity_hash,
                    e.uncased_entity_hash,
                    e.text,
                    e.label,
                    e.processed_at,
                )
                for e in added_entities.values()
            ],
        )

        return out_messages