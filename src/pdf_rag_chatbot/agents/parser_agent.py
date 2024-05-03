from typing import List, Optional

import orjson
from pydantic import BaseModel
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate


class ParserAgent:
	def __init__(self, llm: BaseChatModel):
		self.llm = llm
		self.prompt = ChatPromptTemplate.from_messages([
			(
				"system", 
				"You are a question parser agent that extracts meaningful keywords, phrases, "
				"and entities from user queries to be used for a semantic search operation. "
				"You will always respond using this JSON format:\n\n"
				"{{\"keywords\": [\"keyword1\", \"keyword2\"], \"phrases\": [\"phrase1\", "
				"\"phrase2\"], \"entities\": [\"entity1\", \"entity2\"]}}.\n"
				"Make sure to only include terms the user mentioned in their query. \n\n"
				"Remember that using terse phrases can be more helpful than single keywords "
				"when constructing a semantic search query."
			),
			("user", "{input}"),
		])
		self.chain = self.prompt | self.llm

	def __call__(self, question: str) -> Optional["SearchTerms"]:
		response = self.chain.invoke({"input": question})
		search_terms = SearchTerms(**orjson.loads(response.content))
		
		if not search_terms.keywords and not search_terms.phrases and not search_terms.entities:
			return None
		
		return search_terms


class SearchTerms(BaseModel):
	keywords: List[str]
	phrases: List[str]
	entities: List[str]