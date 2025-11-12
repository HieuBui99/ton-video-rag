
import json
from pathlib import Path

import chromadb
import cv2
import matplotlib.pyplot as plt
import nest_asyncio
import numpy as np
import requests
from tqdm import tqdm
from typing import Literal
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel
from pydantic_ai import Agent, BinaryImage, ImageUrl
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.openai import OpenAIProvider



client = AsyncOpenAI(
    base_url = 'http://localhost:8080/v1',
    api_key='admin', # required, but unused
)

setting = OpenAIChatModelSettings(
    temperature=0.7,
    top_p=0.8,
)
model = OpenAIChatModel("qwem3-vl", provider=OpenAIProvider(openai_client=client))

SYSTEM_PROMPT = """
You are an expert at answering quizes about traffic laws based on images.
When given an image of a vehicle's dashcam, you will answer the quiz question about traffic laws related to the image. You will also be provided with some context about traffic laws in Vietnam to help you answer the question.
The questions will be multiple choice, with options labeled A, B, C, and D. 
You will provide the answer as the letter corresponding to the correct option (A, B, C, or D).
return your answer in the following JSON format:
{"answer": "A", "reasoning": "Your detailed reasoning here."}
Make sure to only return the JSON object, without any additional text
"""

class QuizAnswer(BaseModel):
    answer: Literal['A', 'B', 'C', 'D'] 
    reasoning: str


def get_detailed_instruct(query: str) -> str:
    task = 'Given a web search query, retrieve relevant passages that answer the query'
    return f'Instruct: {task}\nQuery: {query}'

def get_embedding(text: str) -> list[float]:
    response = requests.post(
        "http://localhost:8081/embed",
        headers={"Content-Type": "application/json"},
        data=json.dumps({
            "inputs": [get_detailed_instruct(text)],
        })
    )
    embeddings = response.json()[0]
    return embeddings

def build_prompt(question: str, law_contexts: list[str]) -> str:
    law_context_text = "\n".join([f"- {context}" for context in law_contexts])
    prompt = f"""
    Based on the image and the provided contexts, answer the following question about traffic laws:

    ###Question: 
    {question}

    Provide your answer as the letter corresponding to the correct option (A, B, C, or D).

    ###CONTEXT:
    {law_context_text}
    """
    return prompt

agent = Agent(
    model=model,
    # output_type=[QuizAnswer, ],
    system_prompt=SYSTEM_PROMPT,
    retries=2
)

def main():
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    video_collection = chroma_client.get_collection(name="video_captions_db")
    law_collection = chroma_client.get_collection(name="legal_db")

    with open("train_translated.json", "r") as f:
        question_bank = json.load(f)["data"]
    answers = []
    for query in tqdm(question_bank):
        video_path = query["video_path"]
        question = query["question_en"]

        embedding = get_embedding(question)

        video_results = video_collection.query(
            query_embeddings=[embedding],
            n_results=1,
            include=['metadatas', 'documents'],
            where={"video_path": str(Path(str(Path(video_path).absolute()).replace('/zaloai/', '/zaloai/zaloai/traffic_buddy_train+public_test/')))} # use the absolute video path to filter
        )
        frame_index = video_results['metadatas'][0][0]['frame_index']
        # get the frame from the video
        cap = cv2.VideoCapture(video_results['metadatas'][0][0]['video_path'])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        cap.release()
        # resize the frame to half size
        frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        # encode the frame to jpeg bytes
        success, encoded_image = cv2.imencode('.jpg', frame)
        image_bytes = encoded_image.tobytes()
        image = BinaryImage(data=image_bytes, media_type="image/jpeg")   

        law_results = law_collection.query(
            query_embeddings=[embedding],
            n_results=3,
            include=['documents'],
        )
        law_contexts = [doc for doc in law_results['documents'][0]]

        output = agent.run_sync(
            [
                image,
                build_prompt(question, law_contexts)
            ]
        )
        answers.append(output.output)

    with open("quiz_answers.json", "w") as f:
        json.dump(answers, f, indent=4) 

if __name__ == "__main__":
    # nest_asyncio.apply()
    main()
