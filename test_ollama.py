from openai import AsyncOpenAI
import asyncio

async def main():
    client = AsyncOpenAI(
        base_url="http://host.docker.internal:11434/v1",
        api_key="ollama",
    )

    response = await client.chat.completions.create(
        model="qwen3:8b",
        messages=[
            {
                "role": "user",
                "content": "hello"
            }
        ],
    )

    print(response.choices[0].message.content)

asyncio.run(main())