   

from extractor import MemoryExtractor
from store import MemoryStore

from retriever import MemoryRetriever
from response_processor import process_response 
class MemoryManager:
    def __init__(self, llm_call, embed_fn):
        self.extractor = MemoryExtractor(llm_call)
        self.store = MemoryStore()
        self.retriever = MemoryRetriever(embed_fn)

    def process_input(self, user_input: str):
        extracted = self.extractor.extract(user_input)
        self.store.update(user_input, extracted)

    def get_context(self, query: str):
        memory = self.store.get_all()
        return self.retriever.retrieve(query, memory)
    
    def process_and_store_response(self, response: str):
        """
        Process the model's response to generate a summary and store it in short-term memory.

        Args:
            response (str): The full response text.
        """
        # Generate a concise summary of the response
        response_summary = process_response(response)
        # Update short-term memory with the response summary
        self.store._update_short_term(response_summary)