from fastapi import FastAPI
from src.langchain.components.agent import movie_agent_executor
from src.server.movie_rag_query import MovieQueryInput, MovieQueryOutput
from src.langchain.tests.async_utils import async_retry

app = FastAPI(
    title="Movie Chatbot",
    description="Endpoints for a movie system RAG chatbot",
)


@async_retry(max_retries=3, delay=1)
async def invoke_agent_with_retry(query: str):
    """
    Retry the agents if a tool fails to run. This can help when there
    are intermittent connection issues to external APIs.
    """

    return await movie_agent_executor.ainvoke({"input": query})


@app.get("/")
async def get_status():
    return {"status": "running"}


@app.post("/movie-rag-agent")
async def ask_movie_agent(query: MovieQueryInput) -> MovieQueryOutput:
    print(query.text)
    query_response = await invoke_agent_with_retry(query.text)
    query_response["intermediate_steps"] = [
        str(s) for s in query_response["intermediate_steps"]
    ]

    return query_response
