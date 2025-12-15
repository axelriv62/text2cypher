from pydantic import BaseModel


class MovieQueryInput(BaseModel):
    text: str


class MovieQueryOutput(BaseModel):
    input: str
    output: str
    intermediate_steps: list[str]
