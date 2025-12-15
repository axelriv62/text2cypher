from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
import src.user_consts as consts
from operator import itemgetter


sentiment_template_str = """Your job is to classify the overall sentiment of the provided context
as "negative" or "positive". Even though the review is not classified entirely positive nor negative,
 you have to lean strictly towards one classification.

You are a thoroughly trained machine learning
model that is an expert at sentiment classification.
You diligently complete tasks as instructed.
Don't answer "Mixed" or "Neutral/Mixed".
You never make up any information that isn't there. 
If you don't know an answer, say you don't know.


{context}
"""

sentiment_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["context"],
        template=sentiment_template_str,
    )
)

sentiment_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        template="What is the classification of the prvoded context as 'negative' or 'positive'? Write the answer with one word."
    )
)
messages = [sentiment_system_prompt, sentiment_human_prompt]

sentiment_prompt_template = ChatPromptTemplate(
    input_variables=["context"],
    messages=messages,
)

chat_model = ChatOllama(model=consts.OLLAMA_MODEL, base_url=consts.OLLAMA_BASE_URL, temperature=0)

output_parser = StrOutputParser()

sentiment_chain_ver1 = sentiment_prompt_template | chat_model | output_parser

sentiment_chain_ver2 = ({
        "context": itemgetter("context"),
    }
    | sentiment_prompt_template
    | chat_model
    | StrOutputParser()
)



######################
# TEST
######################
if __name__ == '__main__':
    context = """The medical staff at the hospital were incredibly attentive and 
    supportive during my stay. The facilities were top-notch, making my recovery 
    comfortable and smooth
    """

    print(sentiment_chain_ver1.invoke({"context": context}))

    print(sentiment_chain_ver2.invoke({"context": context}))