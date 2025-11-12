
import json
from tqdm import tqdm
from openai import AsyncOpenAI
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.readers.file import FlatReader
from pathlib import Path


client = AsyncOpenAI(
    base_url = 'http://localhost:8080/v1',
    api_key='admin', # required, but unused
)

model = OpenAIChatModel("qwen", provider=OpenAIProvider(openai_client=client))


agent = Agent(
    model=model,
    system_prompt="You are an expert translation assistant. You are going to be given a prompt and asked to translate it from Vietnamese to English."
)

def main():
    markdown_path = Path("/home/aki/workspace/learning/zaloai/markdown/p2/auto/p2.md")
    md_docs = FlatReader().load_data(markdown_path)
    parser = MarkdownNodeParser()
    nodes = parser.get_nodes_from_documents(md_docs)

    translated_texts = []
    for node in tqdm(nodes, desc=f"Translating {markdown_path}"):
        translation = agent.run_sync(node.text)
        translated_texts.append({
            "original": node.text,
            "translation": translation.output
        })
        
    

    with open("p2.json", "w") as f:
        json.dump(translated_texts, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()