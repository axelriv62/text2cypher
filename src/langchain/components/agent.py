import src.user_consts as consts
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents.format_scratchpad.tools import format_to_tool_messages
from langchain_classic.agents.output_parsers.tools import ToolsAgentOutputParser
from langchain_classic.agents import AgentExecutor
from langchain.tools import tool
import src.tools.tool1_exploration_chroma as exploration_chroma
import src.tools.tool2_exploration_neo4j as exploration_neo4j

agent_chat_model = ChatOllama(model=consts.OLLAMA_MODEL, base_url=consts.OLLAMA_BASE_URL, temperature=0)

@tool
def explore_movie_reviews(question: str) -> str:
    """
    Use this tool to answer questions about WHAT PEOPLE SAID in movie reviews.
    Examples: opinions, critiques, what viewers liked/disliked, acting performances.
    DO NOT use for factual database queries (release dates, ratings, cast lists).
    Answer in French.
    """
    result = exploration_chroma.review_chain.invoke({"question": question})
    return {"result": result}

@tool
def explore_movies_database(question: str) -> str:
    """
    Use this tool to answer FACTUAL questions about movies from the database.
    Examples: release dates, most recent movie, average ratings, actors, directors, genres.
    DO NOT use for review opinions or what people said about movies.
    Answer in French.
    """
    result = exploration_neo4j.explore_movies(question)
    if result:
        return {"result": result}
    return {"result": "Unable to query movies database"}

"""
@tool
def tool3(question: str) -> str:
    result = tool3.similitarity_chain.invoke({"question": question})
    return {"result": result}

@tool
def tool4(question: str) -> str:
    result = tool4.recommandation_chain.invoke({"question": question})
    return {"result": result}
"""

agent_tools = [
    explore_movie_reviews,
    explore_movies_database,
]

agent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful chatbot designed to answer questions
            about movies.
            """,
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent_llm_with_tools = agent_chat_model.bind_tools(agent_tools)

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_tool_messages(
            x["intermediate_steps"]
        ),
    }
    | agent_prompt
    | agent_llm_with_tools
    | ToolsAgentOutputParser()
)

movie_agent_executor = AgentExecutor(
    agent=agent,
    tools=agent_tools,
    return_intermediate_steps=True,
    verbose=True,
)

# TEST

if __name__ == "__main__":
    question = "Summarize what people liked and disliked about the movie Stanley & Iris." # "What is the most recent movie ?"
    response = movie_agent_executor.invoke({"input": question})
    print(response["output"])