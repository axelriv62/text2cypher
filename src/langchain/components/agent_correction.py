import src.user_consts as consts
import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
#from src.langchain.components.rag_sentiment_example_retriever_with_chroma import sentiment_chain
from src.langchain.components.few_shot_prompt_engineering import sentiment_chain

from langchain_classic.agents.format_scratchpad.tools import format_to_tool_messages
from langchain_classic.agents.output_parsers.tools import ToolsAgentOutputParser

from langchain_classic.agents import (
    Tool,
    AgentExecutor,
)
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
    result = review_chain.invoke({"question": question})
    return {"result": result}




@tool
def perform_sentiment_analysis(context: str) -> str:
    """
    Useful for performing sentiment analysis about patient
    experiences. The user asks a question about patient
    experiences and then performs sentiment analysis on the reviews of interest.
    Not useful for answering objective questions that
    involve counting, percentages, aggregations, or listing facts.
    Use the entire prompt as input to the tool. For instance,
    if the prompt is "What is the overall stance regarding the satisfaction of patients' care?",
    the input should be "What is the overall stance regarding the satisfaction of patients' care?".
    """
    result = sentiment_chain.invoke({"context": context})
    return {"result": result}


agent_tools = [
    explore_patient_experiences,
    perform_sentiment_analysis
]







hospital_agent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful chatbot designed to answer questions
            about patient experiences and sentiment analysis.
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
        {"input": "What is the current wait time at hospital Wallace-Hamilton?"}
    )
    print(response.get("output"))

    print("\n\n------------------")

    response = movie_agent_executor.invoke(
            {"input": "What have patients said about their comfort at the hospital?"}
    )
    print(response.get("output"))

    print("\n\n------------------")

    response = movie_agent_executor.invoke(
        {"input": "What is the overall stance of the patients regarding their comfort at the hospital?"}
    )
    print(response.get("output"))
