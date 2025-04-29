from openai import AsyncOpenAI
import asyncio
from typing import List


class EmbeddingOpenAI:
    
    def __init__(self, openai_client: AsyncOpenAI,model:str):
        self.client_async = openai_client
        self.model = model
        
    async def create_embedding(self, input_text: str) -> List[float]:
        try:
            response = await self.client_async.embeddings.create(
                model=self.model,
                input=input_text,
            )
        except Exception as e:
            print(f"Failed to create embedding: {str(e)}")
            print("Trying again after 1 sec")
            await asyncio.sleep(1)
            response = await self.client_async.embeddings.create(
                model=self.model,
                input=input_text,
            )
        return response.data[0].embedding


    async def get_embeddings(self, strings: List[str]) -> List[List[float]]:

        # Create a list of tasks for concurrent execution
        tasks = [self.create_embedding(text) for text in strings]
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        return results
