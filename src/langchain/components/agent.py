import src.user_consts as consts
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents.format_scratchpad.tools import format_to_tool_messages
from langchain_classic.agents.output_parsers.tools import ToolsAgentOutputParser
from langchain_classic.agents import AgentExecutor
from langchain.tools import tool
import src.tools.tool1_exploration_chroma as tool1
import src.tools.tool2_exploration_neo4j as tool2
import src.tools.tool3_similitarity as tool3
import src.tools.tool4_recommandation as tool4

agent_chat_model = ChatOllama(model=consts.OLLAMA_MODEL, base_url=consts.OLLAMA_BASE_URL, temperature=0)

@tool
def tool1(question: str) -> str:
    result = tool1.review_chain.invoke({"question": question})
    return {"result": result}

@tool
def tool2(question: str) -> str:
    result = tool2.explore_movies(question)
    return {"result": result}

@tool
def tool3(question: str) -> str:
    result = tool3.similitarity_chain.invoke({"question": question})
    return {"result": result}

@tool
def tool4(question: str) -> str:
    result = tool4.recommandation_chain.invoke({"question": question})
    return {"result": result}

agent_tools = [
    tool1,
    tool2
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
    question = "What is the most recent movie ?"
    response = movie_agent_executor.invoke({"input": question})
    print(response["output"])