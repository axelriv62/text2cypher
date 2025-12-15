import src.user_consts as consts
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.langchain.components.few_shot_prompt_engineering import sentiment_chain
from langchain_classic.agents.format_scratchpad.tools import format_to_tool_messages
from langchain_classic.agents.output_parsers.tools import ToolsAgentOutputParser
from langchain_classic.agents import AgentExecutor
from langchain.tools import tool



agent_chat_model = ChatOllama(model=consts.OLLAMA_MODEL, base_url=consts.OLLAMA_BASE_URL, temperature=0)

@tool
def explore_patient_experiences(question: str) -> str:
    """
    Useful when you need to answer questions about patient
    experiences, feelings, or any other qualitative question
    that could be answered about a patient using semantic
    search. Not useful for answering objective questions that
    involve counting, percentages, aggregations, or listing facts.
    Use the entire prompt as input to the tool. For instance,
    if the prompt is "Are patients satisfied with their care?",
    the input should be "Are patients satisfied with their care?".
    """


@tool
def explore_patient_sentiments(question: str) -> str:
    result = sentiment_chain.invoke({"question": question})
    return {"result": result}

@tool
def explore_patient_waiting(question: str) -> str:
    return

agent_tools = [
    explore_patient_experiences,
    explore_patient_sentiments
]







hospital_agent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful chatbot designed to answer questions
            about patient experiences.
            """,
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent_llm_with_tools = agent_chat_model.bind_tools(agent_tools)

hospital_agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_tool_messages(
            x["intermediate_steps"]
        ),
    }
    | hospital_agent_prompt
    | agent_llm_with_tools
    | ToolsAgentOutputParser()
)

movie_agent_executor = AgentExecutor(
    agent=hospital_agent,
    tools=agent_tools,
    return_intermediate_steps=True,
    verbose=True,
)



###########################
# TEST
###########################

if __name__ == '__main__':

    response = movie_agent_executor.invoke(
            {"input": "What have patients said about their comfort at the hospital?"}
    )
    print(response.get("output"))
