import os

from operator import itemgetter
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


load_dotenv()

print("Initializing components...")

embeddings = OpenAIEmbeddings(api_key=os.environ.get("OPENAI_API_KEY"), dimensions=1024, model="text-embedding-3-small")
llm = ChatOpenAI()

vectorstore = PineconeVectorStore(index_name=os.environ["INDEX_NAME"], embedding=embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question only based on the following context. If you don't know the answer, say you don't know.
    
    {context}

    Question: {question}

    Provide a detailed answer:
    """
)


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


def retrieval_chain_without_lcel(query: str):
    """
    Simple Retrieval Chain without LCE. It retrieves relevant documents manually and then generates an answer based on those documents.
    """
    # Step 1: Retrieve relevant documents
    docs = retriever.invoke(query)

    # Step 2: Format the retrieved documents into a context string
    context = format_docs(docs)

    # Step 3: Create the prompt by filling in the context and question
    messages = prompt_template.format(context=context, question=query)

    # Step 4: Generate the answer using the LLM
    response = llm.invoke(messages)

    return response.content


def create_retrieval_chain_with_lcel():
    """
    Create a retrieval chain using LCEL.
    Returns a chain that can be invoked with {"question": "..."}
    """
    retrieval_chain = (RunnablePassthrough.assign(context=itemgetter("question") | retriever | format_docs)
        | prompt_template 
        | llm
        | StrOutputParser()
    )

    return retrieval_chain


if __name__ == '__main__':
    print("Retrieving...")

    query = "What is Pinecone in machine learning?"

    print("\n" + "="*50 + "\n")
    # print("IMPLEMENTATION WITHOUT LCEL\n")
    # print("="*50 + "\n")
    # result_without_lcel = retrieval_chain_without_lcel(query)
    print("IMPLEMENTATION WITH LCEL\n")
    print("="*50 + "\n")
    chain_with_lcel = create_retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question": query})

    print("Answer with LCEL:")
    print(result_with_lcel)