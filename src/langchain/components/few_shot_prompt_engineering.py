from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
import src.user_consts as consts
from operator import itemgetter




sentiment_template_str = """
Task:
Your job is to classify the overall sentiment of the reviews
as "negative" and "positive".

<article>{context}</article>

Instructions:
You are a thoroughly trained machine learning
model that is an expert at sentiment classification.
You diligently complete tasks as instructed.

Note:
You never make up any information that isn't there. 
If you don't know an answer, say you don't know.
Don't answer "Mixed".
Even though the review is neither entirely positive nor negative, 
you have to lean slightly towards one classification.

Examples:
# <article>The medical staff at the hospital were incredibly attentive and 
    supportive during my stay. The facilities were top-notch, making my recovery 
    comfortable and smooth.</article>
Output: "positive"

# <article>The hospital's commitment to patient education impressed me. 
    The medical team took the time to explain my diagnosis and treatment options,
    empowering me to make informed decisions about my health.</article>
Output: "positive"

# <article>My time at the hospital was challenging. The nurses seemed overwhelmed,
    and communication between the staff and patients was lacking. Improvement is 
    needed in patient care.</article>
Output: "negative"

# <article>I was disappointed by the lack of follow-up care after my discharge. 
    It left me feeling unsupported in my recovery journey. Comprehensive post-discharge care 
    is vital for patients' overall well-being.</article>
Output: "negative"
"""

sentiment_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["context"],
        template=sentiment_template_str,
    )
)

sentiment_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        template="What is the classification of the review as 'negative' and 'positive'? Write the answer with one word."
    )
)
messages = [sentiment_system_prompt, sentiment_human_prompt]

sentiment_prompt_template = ChatPromptTemplate(
    input_variables=["context"],
    messages=messages,
)

chat_model = ChatOllama(model=consts.OLLAMA_MODEL, base_url=consts.OLLAMA_BASE_URL, temperature=0)

output_parser = StrOutputParser()

sentiment_chain = ({
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
    context = "I appreciated the thorough care provided by the hospital staff. The facilities were comfortable, but the cafeteria food was subpar."
    print(sentiment_chain.invoke({"context": context}))

    context = "I had a positive experience overall at the hospital. The medical staff was attentive, and the facilities were well-maintained. However, the Wi-Fi was unreliable."
    print(sentiment_chain.invoke({"context": context}))

    context = "I had a mixed experience at the hospital. The doctors were skilled, but the administrative processes were confusing, leading to frustration and delays."
    print(sentiment_chain.invoke({"context": context}))
