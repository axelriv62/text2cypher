from src.langchain.components.utils_llm import CustomCallback
from src.langchain.components.sentiment_chain import sentiment_chain_ver2



######################
# TEST
######################
if __name__ == '__main__':
    handler = CustomCallback()

    context = "I appreciated the thorough care provided by the hospital staff. The facilities were comfortable, but the cafeteria food was subpar."
    result = sentiment_chain_ver2.invoke({"context": context}, {"callbacks": [handler]})
    print("result:", result)


    for i,x in enumerate(handler.kwargs):
        print("--------------------")
        print(x)
        print(handler.chain_output[i])