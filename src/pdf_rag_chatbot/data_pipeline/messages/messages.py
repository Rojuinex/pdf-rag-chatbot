import uuid
from typing import Optional, Union

from pydantic import BaseModel, Field

from pdf_rag_chatbot.db.models import (
	Document,
	Sentence,
	Entity,
)

class Message(BaseModel):
	message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

class DeadLetterMessage(Message):
	step: str
	error: str
	traceback: str
	request: Message

class FileUploaded(Message):
	file_path: str
	session_id: Optional[str] = None

class DocumentCreated(Message):
	document: Document

class SentenceCreated(Message):
	sentence: Sentence

class EntityCreated(Message):
	entity: Entity