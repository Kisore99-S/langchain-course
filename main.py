from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
# from tavily import TavilyClient
from langchain_tavily import TavilySearch
from dotenv import load_dotenv

load_dotenv()


# tavily = TavilyClient()


# @tool
# def search(query: str) -> str:
#     """
#         Tool that searches the web for the given query and returns the results as a string.
#         Args: query (str): The search query.
#         Returns: str: The search results.
#     """
#     print(f"Searching for: {query}")
#     return tavily.search(query=query)


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)


def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": HumanMessage(content="Search for 3 job positings for AI engineer using langchain in Hyderabad from LinkedIn and list their details.")})
    print(result)


if __name__ == "__main__":
    main()
