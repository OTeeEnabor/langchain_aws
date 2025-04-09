import pprint
import sys
import os
import uuid
from pathlib import Path
import pandas as pd

import redis
import openai
from langchain.llms import OpenAI
from langchain.utilities import SQLDatabase, SerpAPIWrapper
from langchain.memory.chat_message_histories import RedisChatMessageHistory
from langchain_experimental.sql import SQLDatabaseChain
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.memory.chat_message_histories import SQLChatMessageHistory
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
import openai

from dotenv import load_dotenv

load_dotenv()

serpapi_key = os.getenv("SERP_API_KEY")

def initialize_agent(api_key, db_name):
    api_key = os.getenv("OPENAI_API_KEY")
    openai.api_key = api_key

    # initialize Redis memory
    session_id = str(uuid.uuid4())
    redis_url = os.getenv("REDIS_URL")
    message_history = RedisChatMessageHistory(url=redis_url, ttl=600, session_id=session_id)
    # initialize conversation memory
    memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=message_history)

    # initialize the SerpApi Wrapper with your API key
    search = SerpAPIWrapper(serpapi_api_key=serpapi_key)
    serapi_tool = Tool(name="Search", func=search.run, description="useful for when you need to answer questions about the yield curve, finance or economics")

    # initialize databse
    # define the connection details for SQL Database
    db_user = os.getenv("user")
    db_password = os.getenv("password")
    db_host = os.getenv("host")
    db_name = os.getenv("database")

    # create DB connection string
    db_uri = f"postgresql://{db_user}:{db_password}@{db_host}/{db_name}"

    db = SQLDatabase.from_uri(db_uri) # error handle

    # Define few-shot examples specific to chosen database
    few_shots = {
        "What was the highest unemployment rate last year?": "SELECT MAX(UNRATE) FROM economic_indicators WHERE Date >= '2022-01-01 00:00:00' AND Date <= '2022-12-31 00:00:00';",
        "Average industrial production for the previous month?": "SELECT AVG(IPN213111S) FROM production_data WHERE Date >= date('now', 'start of month', '-1 month', '00:00:00') AND Date < date('now', 'start of month', '00:00:00');",
        "Show the five lowest 10-year yield rates of the current year.": "SELECT DGS10 FROM yield_curve_prices WHERE Date >= '2023-01-01 00:00:00' ORDER BY DGS10 ASC LIMIT 5;",
        "List the economic indicators for the first quarter of 2023.": "SELECT * FROM economic_indicators WHERE Date >= '2023-01-01 00:00:00' AND Date <= '2023-03-31 00:00:00';",
        "What are the latest available production numbers for Saudi Arabia?": "SELECT SAUNGDPMOMBD FROM production_data WHERE Date = (SELECT MAX(Date) FROM production_data);",
        "Compare the unemployment rate at the beginning and end of the last recession.": "SELECT UNRATE FROM economic_indicators WHERE Date IN (SELECT Start_Date FROM business_cycles WHERE Phase = 'Contraction' ORDER BY Start_Date DESC LIMIT 1) OR Date IN (SELECT End_Date FROM business_cycles WHERE Phase = 'Contraction' ORDER BY End_Date DESC LIMIT 1);",
        "Find the average civilian labor force participation rate for the last year.": "SELECT AVG(CIVPART) FROM economic_indicators WHERE Date >= '2022-01-01 00:00:00' AND Date <= '2022-12-31 00:00:00';",
        "Show the change in 2-year yield rates over the past six months.": "SELECT DGS2 FROM yield_curve_prices WHERE Date >= date('now', '-6 months', '00:00:00') ORDER BY Date;",
        "What was the maximum production of natural gas in Qatar last year?": "SELECT MAX(QATNGDPMOMBD) FROM production_data WHERE Date >= '2022-01-01 00:00:00' AND Date <= '2022-12-31 00:00:00';",
        "List the top 3 longest economic expansions since 2000.": "SELECT Start_Date, End_Date FROM business_cycles WHERE Phase = 'Expansion' AND Start_Date >= '2000-01-01 00:00:00' ORDER BY (julianday(End_Date) - julianday(Start_Date)) DESC LIMIT 3;",
    }

    # create a retriever for few shot examples embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    # for each zero shot question key and sql query value store them as langchain Document
    few_shot_docs = [
        Document(page_content=question, metadata={"sql_query": few_shots[question]})
        for question in few_shots.keys()
    ]

    # create a vector DB using the langchain docs and embeddings
    vector_db = FAISS.from_documents(few_shot_docs, embeddings)

    # create a vector store retriever
    retriever = vector_db.as_retriever()

    # create retriever tool for getting similar examples
    retriever_tool = create_retriever_tool(
        retriever,
        name="sql_get_similar_examples",
        description="Retrieves similar sql examples.",
    )

    # define the SQL tool
    sql_tool = Tool(name="SQL", func=retriever_tool.run, description="useful for querying financial data from the database")

    # add bot the SerpAPI and SQL tools to your list of tools
    tools = [serapi_tool, sql_tool]

    # use zeroshot agent to create the prompt
    prefix = """Have a conversation with a human, answering the following questions as best as you can. You have access to the following tools:"""
    suffix="""Begin!\n\n{chat_history}\nQuestion: {input}\n {agent_scratchpad}"""
    prompt = ZeroShotAgent.create_prompt(tools=tools, prefix=prefix, suffix=suffix, input_variables=["input", "chat_history", "agent_scratchpad"])

    # initialize the Langchain LLM with OpenAI
    llm = OpenAI(openai_api_key=api_key, temperature=0, verbose=True)
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
    agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory)


    return agent_chain