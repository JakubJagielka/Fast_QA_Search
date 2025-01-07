from openai import AsyncOpenAI

class LLM_QA:
    
    def __init__(self, client: AsyncOpenAI):
        self.client = client
    
    async def answer(self, query: str, context: list[str], model = 'gpt-4o-mini') -> str:
        context = "\n\n".join(context)
        response = await self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Answer only based on the following context and knowledge you are sure about."},
                {"role": "user", "content": "Context: " + context + "\n\nQuestion: " + query},
            ]
        )
        
        return response.choices[0].message.content