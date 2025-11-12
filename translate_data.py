
import json
from tqdm import tqdm
from openai import AsyncOpenAI
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider



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
    with open("/media/tien/SSD-NOT-OS/hieu_workspace/zaloai/zaloai/traffic_buddy_train+public_test/train/train.json", "r") as f:
        data = json.load(f)
    for item in tqdm(data['data']):
        question = item['question']
        translation = agent.run_sync(question)
        item['question_en'] = translation.output

        choices_en = []
        for choice in item['choices']:
            choice_translation = agent.run_sync(choice)
            choices_en.append(choice_translation.output)
        item['choices_en'] = choices_en

    with open("train_translated.json", "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()