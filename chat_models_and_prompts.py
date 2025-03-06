import os
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate


''' initialize '''

load_dotenv()

model = init_chat_model("llama-3.3-70b-versatile", model_provider="groq")






''' set prompts '''

# method 1
messages = [
    SystemMessage("Translate the following from English into Korean"),
    HumanMessage("hi!"),
]

# method 2
system_template = "Translate the following from English into {language}"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)
prompt = prompt_template.invoke({"language": "Italian", "text": "hi!"})






''' call a LLM model '''
# method 1
# print(model.invoke(messages))

# method 2
# for token in model.stream(messages):
#     print(token.content, end="|")

# method 3
response = model.invoke(prompt)
print(response.content)
