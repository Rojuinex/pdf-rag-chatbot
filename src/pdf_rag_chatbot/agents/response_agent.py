from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate


class ResponseAgent:
	def __init__(self, llm: BaseChatModel):
		self.llm = llm
		self.prompt = ChatPromptTemplate.from_messages([
			(
				"system", 
				"You are an intelligent AI agent that helps people answer questions "
				"from documents they have uploaded. You will receive the user's question "
				"and a list of search results from a semantic search operation.\n\n"
				"Your goal is to analyze the user's question and search results to provide "
				"the most accurate and relevant information to the user."
			),
			("user", "Search results: {search_results}\n\nQuestion: {question}"),
		])
		self.chain = self.prompt | self.llm

	def __call__(self, question: str, search_results: str) -> str:
		response = self.chain.invoke({"question": question, "search_results": search_results})
		return response.content
