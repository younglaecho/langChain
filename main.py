import os

from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import BaseOutputParser


class CommaSeparatedListOutputParser(BaseOutputParser):
    """Parse the output of an LLM call to a comma-separated list."""

    def parse(self, text: str):
        """Parse the output of an LLM call."""
        return text.strip().split(", ")

def invokeTest():
    llm = OpenAI()
    chat_model = ChatOpenAI()

    text = "What would be a good company name for a company that makes colorful socks?"
    messages = [HumanMessage(content=text)]

    print(llm.invoke(text))
    print(chat_model.invoke(messages))

def promptTemplateTest():

    prompt = PromptTemplate.from_template("What would be a good company name for a company that makes {product}?")
    prompt.format(product="colorful socks")

    template = "You are a helpful assistant that translates {input_language} to {output_language}."
    human_template = "{text}"

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        ("human", human_template),
    ])

    chat_list = chat_prompt.format_messages(input_language="English", output_language="French", text="I love programming.")
    llm = OpenAI()
    chat_model = ChatOpenAI()

    print(llm.invoke(chat_list))
    print(chat_model.invoke(chat_list))

    print(type(chat_prompt.invoke({"input_language":"English", "output_language":"French", "text":"I love programming."})))
    print(type(chat_prompt.invoke({"input_language":"English", "output_language":"French", "text":"I love programming."}).to_messages()))
    print(type(chat_prompt.invoke({"input_language":"English", "output_language":"French", "text":"I love programming."}).to_string()))

if __name__ == '__main__':
    load_dotenv()

    # invokeTest()
    # promptTemplateTest()

    CommaSeparatedListOutputParser().parse("hi, bye")

    template = """You are a helpful assistant who generates comma separated lists.
    A user will pass in a category, and you should generate 5 objects in that category in a comma separated list.
    ONLY return a comma separated list, and nothing more."""
    human_template = "{text}"

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        ("human", human_template),
    ])
    chain = chat_prompt | ChatOpenAI() | CommaSeparatedListOutputParser()

    print(chain.invoke({"text": "colors"}))
