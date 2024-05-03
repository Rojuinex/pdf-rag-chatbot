import uuid
from typing import List, Dict, Tuple

import torch
import duckdb
import gradio as gr
import polars as pl
from  sentence_transformers.util import cos_sim
from langchain_core.language_models import BaseLLM
from loguru import logger

from pdf_rag_chatbot.agents.parser_agent import SearchTerms
from pdf_rag_chatbot.db import setup_database
from pdf_rag_chatbot.data_pipeline import TextPipeline
from pdf_rag_chatbot.data_pipeline.messages import FileUploaded
from pdf_rag_chatbot.agents import (
    ParserAgent,
    ResponseAgent,
)


class App:
    def __init__(self, database: str, llm: BaseLLM):
        """Initialize the app.


        Args:
            database (str): The path to the DuckDB database file.

        Raises:
            Exception: If the database connection fails.
        """
        self.db = duckdb.connect(database)
        setup_database(self.db)

        self.text_pipeline = TextPipeline(self.db)

        self.llm = llm
        self.parser_agent =  ParserAgent(llm=llm)
        self.response_agent = ResponseAgent(llm=llm)


    def __del__(self):
        """Close the database connection."""
        logger.debug("Closing database connection.")
        self.db.close()
    
    def handle_message(
        self,
        session_id: str,
        message: Dict,
        messages: List[Dict],
    ):
        """Handle a message from the user.

        Args:
            message (str): The message from the user.
            history (List[str]): The history of messages from the user.
            files (Optional[str | List[str]], optional): The files uploaded by the user. Defaults to None.
        """
        try:
            files = message.get("files", None)
            message_text = message["text"]

            if files:
                messages.append({ "role": "file_upload", "files": files })
                messages.append({ "role": "assistant", "text": "📑 Processing uploaded files..." })
                yield {"text": "", "files": []}, messages, self.raw_history_to_chatbot(messages)

                for file in files:
                    self.text_pipeline(
                        FileUploaded(file_path=file, session_id=session_id)
                    )
                messages = messages[:-1]

            messages.append({ "role": "user", "text": message_text })

            if message_text == "":
                messages.append({ "role": "assistant", "text": "It seems you haven't asked a question." })
                yield {"text": "", "files": []}, messages, self.raw_history_to_chatbot(messages)
                return

            messages.append({"role": "assistant", "text": "🤖 Analyzing your question..."})
            yield {"text": "", "files": []}, messages, self.raw_history_to_chatbot(messages)

            search_terms = self.parser_agent(message_text)
            logger.debug(f"Extracted search terms: {search_terms}")

            if search_terms is not None:
                search_terms.phrases.append(message_text)
            else:
                search_terms = SearchTerms(keywords=[], phrases=[message_text], entities=[])


            messages = messages[:-1]
            messages.append({"role": "assistant", "text": "🔍 Searching through documents"})
            yield {"text": "", "files": []}, messages, self.raw_history_to_chatbot(messages)

            search_results = self.search_documents(session_id, search_terms)

            messages = messages[:-1]
            messages.append({"role": "assistant", "text": "💬 Preparing response..."})
            yield {"text": "", "files": []}, messages, self.raw_history_to_chatbot(messages)

            response = self.response_agent(message_text, search_results)

            messages = messages[:-1]
            messages.append({
                "role": "assistant",
                "text": response
            })

            yield {"text": "", "files": []}, messages, self.raw_history_to_chatbot(messages)
        except Exception as e:
            logger.exception(e)
            messages.append({
                "role": "assistant",
                "text": f"I seem to have encountered an error while attempting to answer your question. Please try again." 
            })
            yield {"text": "", "files": []}, messages, self.raw_history_to_chatbot(messages)

    def raw_history_to_chatbot(self, hist : List[Dict]) -> List[Tuple[str, str]]:
        disp_hist = []
        current_pair = None
        for h in hist:
            if h["role"] == "file_upload":
                for file in h["files"]:
                    disp_hist.append((file, None))
            elif h["role"] == "user":
                current_pair = (h["text"], None)
            elif h["role"] == "assistant":
                if current_pair is None:
                    current_pair = (None, None)
                current_pair = (current_pair[0], h["text"])
                disp_hist.append(current_pair)
                current_pair = None

        if current_pair is not None:
            disp_hist.append(current_pair)

        return disp_hist


    def clear_history(self):
        """Clear the chat history.

        Returns:
            Tuple[str, Dict, List, List]: The initial state of the chat.
        """
        return str(uuid.uuid4()), {"text": "", "files": []}, [], []

    def launch(self, *args, **kwargs):
        with gr.Blocks() as app:
            session_id = gr.State(str(uuid.uuid4()))
            raw_history = gr.State([])

            chatbot = gr.Chatbot(label="Chatbot")
            clear = gr.ClearButton()

            msg = gr.MultimodalTextbox(
                show_label=False,
                label="Message",
                placeholder="Type your message here...",
                autofocus=True,
                file_types=[".pdf", ".txt", ".text"],
            )

            clear.click(self.clear_history, [], [session_id, msg, raw_history, chatbot])
            msg.submit(self.handle_message, [session_id, msg, raw_history], [msg, raw_history, chatbot])

        app.launch(*args, **kwargs)

    def search_documents(
            self,
            session_id: str,
            search_terms: SearchTerms,
            entity_importance: float = 0.6,
            document_context_size: int = 3
        ):
        sentences_df = None

        if search_terms.keywords or search_terms.phrases:
            term_embeddings = self.text_pipeline.embed.model.encode(
                search_terms.keywords + search_terms.phrases,
                convert_to_tensor=True
            ).to(self.text_pipeline.embed.device)

            sentences_df = self.db.execute(
                """--sql
                    SELECT
                        cased_text_hash as cased_sentence_hash,
                        embedding
                    FROM text_embedding
                    WHERE
                        cased_text_hash IN (
                            SELECT DISTINCT
                                cased_sentence_hash
                            FROM document_sentence ds
                            JOIN uploaded_file uf USING(document_hash)
                            WHERE
                                session_id = ?
                                OR session_id IS NULL
                        )
                        AND model_name = ?
                """,
                (session_id, self.text_pipeline.embed.model_name)
            ).pl()

            sentence_embeddings = (
                torch.tensor(sentences_df["embedding"].to_list())
                .to(self.text_pipeline.embed.device)
            )
            sentences_df = sentences_df.drop("embedding")
            cos_scores = cos_sim(term_embeddings, sentence_embeddings)

            cos_scores = cos_scores.cpu().numpy()
            sentences_df = (
                sentences_df.with_columns(
                    score=cos_scores.max(axis=0)
                )
                .sort(by="score", descending=True)
                .slice(0, 100)
            )

            logger.debug(f"Sentences: {sentences_df}")

            del term_embeddings
            del sentence_embeddings
            del cos_scores

        if search_terms.entities:
            term_embeddings = self.text_pipeline.embed.model.encode(
                search_terms.entities,
                convert_to_tensor=True
            ).to(self.text_pipeline.embed.device)
            entities_df = self.db.execute(
                """--sql
                    SELECT
                        cased_text_hash as cased_entity_hash,
                        embedding
                    FROM text_embedding
                    WHERE
                        cased_text_hash IN (
                            SELECT DISTINCT
                                cased_entity_hash
                            FROM document_entity de
                            JOIN uploaded_file uf USING(document_hash)
                            WHERE
                                session_id = ?
                                OR session_id IS NULL
                        )
                        AND model_name = ?
                """,
                (session_id, self.text_pipeline.embed.model_name)
            ).pl()
            entity_embeddings = (
                torch.tensor(entities_df["embedding"].to_list())
                .to(self.text_pipeline.embed.device)
            )
            entities_df = entities_df.drop("embedding")
            cos_scores = cos_sim(term_embeddings, entity_embeddings)
            cos_scores = cos_scores.cpu().numpy()
            entities_df = (
                entities_df.with_columns(
                    entity_score=cos_scores.max(axis=0)
                )
                .sort(by="entity_score", descending=True)
                .slice(0, 100)
            )
            logger.debug(f"Entities: {entities_df}")
            del term_embeddings
            del entity_embeddings
            del cos_scores

            entity_sentence_df = self.db.execute(
                """--sql
                    SELECT
                        cased_sentence_hash,
                        MAX(entity_score) AS score
                    FROM entities_df
                    JOIN document_entity de USING(cased_entity_hash)
                    JOIN document_sentence ds
                        ON de.document_hash = ds.document_hash
                        AND de.sentence_index = ds.index
                    GROUP BY cased_sentence_hash
                """,
            ).pl()

            if sentences_df is None:
                sentences_df = entity_sentence_df
            else:
                # merge sentence scores
                sentences_df = (
                    sentences_df.select(
                        "cased_sentence_hash",
                        pl.col("score").alias("score_a")
                    )
                    .join(
                        entity_sentence_df.select(
                            "cased_sentence_hash",
                            pl.col("score").alias("score_b")
                        )
                        .unique(),
                        on="cased_sentence_hash",
                        how="outer"
                    )
                    .fill_null(strategy="zero")
                    .with_columns(
                        score=(
                            (1 - entity_importance) * pl.col("score_a")
                            + (entity_importance * pl.col("score_b"))
                        ),
                    )
                    .drop(["score_a", "score_b"])
                )
        # end if search_terms.entities

        sentences_df = sentences_df.sort(by="score", descending=True).slice(0, 50)


        search_results = self.db.execute(
            """--sql
                SELECT DISTINCT
                    -- document_hash,
                    -- cased_sentence_hash,
                    text AS sentence_text,
                    score AS relevancy_score,
                    -- "index" AS sentence_index,
                    (
                        SELECT
                            STRING_AGG(text, ' ' ORDER BY "index") AS text
                        FROM document_sentence ds_inner
                        WHERE
                            ds_inner.document_hash = ds.document_hash
                            AND ds_inner."index" BETWEEN ds."index" - $ctx_size AND ds."index" + $ctx_size
                    ) AS surrounding_context
                FROM document_sentence ds
                JOIN sentences_df USING(cased_sentence_hash)
                JOIN uploaded_file uf USING(document_hash)
                WHERE
                    session_id = $session_id
                    OR session_id IS NULL
                ORDER BY score DESC
            """,
            {
                "session_id": session_id,
                "ctx_size": document_context_size
            }
        ).pl()

        logger.debug(f"Search results: {search_results}")

        return search_results.write_json()


