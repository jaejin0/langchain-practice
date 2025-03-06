import os
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

model = init_chat_model("llama-3.3-70b-versatile", model_provider="groq")

messages = [
    SystemMessage("Translate the following from English into Korean"),
    HumanMessage("hi!"),
]

print(model.invoke(messages))


for token in model.stream(messages):
    print(token.content, end="|")
