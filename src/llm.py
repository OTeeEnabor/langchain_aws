import os
from pathlib import Path
import pprint
import sys
import pandas as pd
from langchain.llms import OpenAI
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.agents.agent_toolkits import create_retriever_tool
import openai

from dotenv import load_dotenv

PROJECT_PATH = Path.cwd().parent

sys.path.append(str(PROJECT_PATH))

from config.config_ import load_config
from config.connect import connect

load_dotenv()


# Set up OpenAI API Key
API_KEY = os.getenv("OPEN_AI_API_KEY")
OpenAI.api_key = API_KEY

# initialize SQL database
# define the connection details for SQL Database
db_user = os.getenv("user")
db_password = os.getenv("password")
db_host = os.getenv("host")
db_name = os.getenv("database")

# create DB connection string
db_uri = f"postgresql://{db_user}:{db_password}@{db_host}/{db_name}"

# connect to the DB
db = SQLDatabase.from_uri(db_uri)

# instantiate OpenAI model
llm = OpenAI(openai_api_key=API_KEY, temperature=0.2, verbose=True)

#
# Create a chain to interact with SQL database using LLM model and SQL DB
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

# User Query
user_query = "Describe the median short rates during the last year period."

# Execute the chain - i.e sequence of actions to interact with SQL database based on user query
response = db_chain.run(user_query)

pprint.pprint(response)

# SQL Agents

"""
Initializes an SQL agent. And creates an agent with the specified properties
trough_month
start_date
end_date
"""

agent_executor = create_sql_agent(

    llm=llm,
    toolkit=SQLDatabaseToolkit(db=db, llm=llm),
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    top_k=10,
)

# create sample query to query the SQL agent
query_result = agent_executor.run(
    "Discuss the yield curve, and their average values over the last economic cycle."
)

print("Query result:", query_result)


# Define few-shot examples specific to your database
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

# create embeddings
embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)
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

# create agent and agent executor that uses retriever tool
toolkit = SQLDatabaseToolkit(
    db=db,
    llm=llm,
)
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_  type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    extra_tools=[retriever_tool],
    top_k=10,
)

# create an example query using the agent 
query_result = agent_executor.run("""Discuss the yield curve, their average values, over the lass economic cycle.""")

print("Query result:", query_result)


# perform description of table task - does agent understand the schema of database

