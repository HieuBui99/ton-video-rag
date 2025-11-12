import asyncio
import json
import os
import uuid
import aiohttp
import chromadb
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def chunked(iterable, size):
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

# Each query must come with a one-sentence instruction that describes the task
task = 'Given a web search query, retrieve relevant passages that answer the query'

async def vectorize_texts(texts, bs=8):
    timeout = aiohttp.ClientTimeout(total=30)
    retries = 3
    results = []

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for batch in tqdm(chunked(texts, bs)):
            for attempt in range(retries):
                try:
                    payload = {"inputs": list(batch)}
                    result = await session.post(
                        "http://localhost:8081/embed",
                        data=json.dumps(payload),
                        headers={"Content-Type": "application/json"},
                    )
                    
                    result = await result.json()
                    results.extend(result)
                    break
                except Exception as e:
                    print(e, batch)
                    if attempt < retries - 1:
                        await asyncio.sleep(1)
                    else:
                        raise e

    return results


async def main():
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="legal_db")
    # glob json files in current directory
    data_path = "p3.json"
    df = pd.read_json(data_path)
    texts = df["translation"].tolist()

    vectors = await vectorize_texts(texts, bs=1)

    collection.add(
        documents=texts,
        embeddings=vectors,
        ids=[str(uuid.uuid4()) for _ in range(len(texts))],
    )
    



if __name__ == "__main__":
    asyncio.run(main())