from langchain_core.documents import Document
from langchain_core.callbacks import BaseCallbackHandler
from typing import Any


# source: https://medium.com/data-science/callbacks-and-pipeline-structures-in-langchain-925aa077227e
class CustomCallback(BaseCallbackHandler):
    def __init__(self):
        self.chain_output = []
        self.kwargs = []

    def on_chain_end(self, outputs: dict[str, Any] | Any, **kwargs: Any) -> None:
        self.chain_output.append(outputs)
        self.kwargs.append(kwargs)

def format_retrieved_documents(documents: list[Document]) -> str:
    """Format retrieved documents and metadata as a single string"""

    result = ""
    #print(len(documents))
    for doc in documents:
        #print("--")
        #print(doc.page_content)

        # question = doc.page_content.strip().replace('question: ', '')
        page_content = doc.page_content
        result += f"{page_content}\n"

        for key in doc.metadata.keys():
            result += f"{key}:\n{doc.metadata[key]}"

        result += "\n\n"

    return result