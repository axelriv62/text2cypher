import src.user_consts as consts
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents.format_scratchpad.tools import format_to_tool_messages
from langchain_classic.agents.output_parsers.tools import ToolsAgentOutputParser
from langchain_classic.agents import AgentExecutor
from langchain.tools import tool
import src.tools.tool1_exploration_chroma as exploration_chroma
import src.tools.tool2_exploration_neo4j as exploration_neo4j
import src.tools.tool3_similitarity as similarity_tool
import src.tools.tool4_recommandation as recommandation_tool

agent_chat_model = ChatOllama(model=consts.OLLAMA_MODEL, base_url=consts.OLLAMA_BASE_URL, temperature=0)

@tool
def explore_movie_reviews(question: str) -> str:
    """
    Use this tool to answer questions about WHAT PEOPLE SAID in movie reviews.
    Examples: opinions, critiques, what viewers liked/disliked, acting performances.
    DO NOT use for factual database queries (release dates, ratings, cast lists).
    """
    result = exploration_chroma.review_chain.invoke({"question": question})
    return {"result": result}

@tool
def explore_movies_database(question: str) -> str:
    """
    Use this tool to answer FACTUAL questions about movies from the database.
    Examples: release dates, most recent movie, average ratings, actors, directors, genres.
    DO NOT use for review opinions or what people said about movies.
    """
    result = exploration_neo4j.explore_movies(question)
    if result:
        return {"result": result}
    return {"result": "Unable to query movies database"}


@tool
def similarity_users(concept: str) -> str:
    """
    Use this tool to compute similarity between two users based on their movie preferences.
    The input should be two usernames separated by a comma.
    Example input: "User1, User2"
    The output is a similarity score between 0 and 1.
    0 means no similarity, 1 means identical preferences.
    """
    result = similarity_tool.get_similarity(concept)
    return {"result": result}

@tool
def recommendation(concept: str) -> str:
    """
    Use this tool to get movie recommendations based on a concept (genre, actor, or director).
    The input is the name of the concept.
    The output is a list of recommended movies with details.
    """
    result = recommandation_tool.get_recommendation(concept)
    return {"result": result}

agent_tools = [
    explore_movie_reviews,
    explore_movies_database,
    similarity_users,
    recommendation,
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

######################
# TEST
######################

if __name__ == "__main__":
    questions = [
        "Summarize what people liked and disliked about the movie Stanley & Iris.", # Test pour l'outil 1
        "What is the most recent movie ?", # Test pour l'outil 2
        "What is the similarity between Sean Mendez and David Malone", # Test pour l'outil 3
        "Give me recommandations on Peter Sollett", # Test pour l'outil 4
    ]
    response = movie_agent_executor.invoke({"input": questions[2]})
    print(response["output"])