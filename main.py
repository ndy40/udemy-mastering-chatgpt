import dotenv
from langchain.chains import LLMChain
from langchain.memory import ConversationSummaryMemory
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

dotenv.load_dotenv()

chat = ChatOpenAI(verbose=True)

prompt = ChatPromptTemplate(
    input_variables=["content", "messages"],
    messages=[
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template("{content}"),
    ],
)

memory = ConversationSummaryMemory(
    memory_key="messages", return_messages=True, llm=chat
)
chain = LLMChain(llm=chat, prompt=prompt, memory=memory, verbose=True)

while True:
    content = input(">> ")
    result = chain({"content": content})
    print(result["text"])