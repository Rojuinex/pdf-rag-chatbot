from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel

class UploadedFile(BaseModel):
	file_uuid: str
	file_path: str
	document_hash: str
	session_id: Optional[str] = None
	uploaded_at: datetime = datetime.now()

class Document(BaseModel):
	document_hash: str
	text: str
	processed_at: datetime = datetime.now()

class Sentence(BaseModel):
	cased_sentence_hash: str
	uncased_sentence_hash: str
	text: str
	processed_at: datetime = datetime.now()

class Entity(BaseModel):
	cased_entity_hash: str
	uncased_entity_hash: str
	text: str
	label: str
	processed_at: datetime = datetime.now()

class DocumentSentence(BaseModel):
	document_hash: str
	cased_sentence_hash: str
	uncased_sentence_hash: str
	text: str
	index: int
	start_char: int
	end_char: int
	processed_at: datetime = datetime.now()

class DocumentEntity(BaseModel):
	document_hash: str
	cased_entity_hash: str
	uncased_entity_hash: str
	text: str
	sentence_index: int
	start_char: int
	end_char: int
	label: str
	processed_at: datetime = datetime.now()