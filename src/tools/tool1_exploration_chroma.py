import os
import pandas as pd

from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings

from langchain_core.documents import Document
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from src.user_consts import IN_FOLDER

import src.user_consts as consts

REVIEWS_CSV_PATH = os.path.join(IN_FOLDER, "movie/movie_reviews.csv")
REVIEWS_CHROMA_PATH = os.path.join(consts.OUT_FOLDER, "chroma_movie_reviews")

review_template_str = """You are an assistant that explores and summarizes movie reviews.

Use ONLY the following review excerpts as context to answer the user question.
- Do not invent facts not present in the reviews.
- If the reviews do not contain enough information, say you don't know.
- When possible, synthesize common points (positives/negatives) mentioned by multiple reviewers.
- Answer in French

Context (movie reviews):
{context}
"""

review_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["context"],
        template=review_template_str,
    )
)

review_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["question"],
        template="{question}",
    )
)

review_prompt_template = ChatPromptTemplate(
    input_variables=["context", "question"],
    messages=[review_system_prompt, review_human_prompt],
)


def format_reviews(docs):
    formatted = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata or {}
        movie_name = meta.get("movie_name", "unknown")
        rating = meta.get("movie_rating", "unknown")
        sentiment = meta.get("review_sentiment", "unknown")

        formatted.append(f"Review {i} | movie={movie_name} | rating={rating} | sentiment={sentiment}")
        formatted.append(doc.page_content.strip())
        formatted.append("")  # ligne vide
    return "\n".join(formatted).strip()


def build_or_load_reviews_db():
    embeddings = OllamaEmbeddings(model=consts.OLLAMA_MODEL, base_url=consts.OLLAMA_BASE_URL)

    if os.path.exists(REVIEWS_CHROMA_PATH) and len(os.listdir(REVIEWS_CHROMA_PATH)) > 0:
        return Chroma(
            persist_directory=REVIEWS_CHROMA_PATH,
            embedding_function=embeddings,
        )
    df = pd.read_csv(REVIEWS_CSV_PATH)

    required_cols = {"movie_review", "movie_name", "movie_rating", "review_sentiment", "movie_url"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"movie_reviews.csv invalide, colonnes manquantes: {sorted(missing)}")

    docs = []
    for _, row in df.iterrows():
        review_text = str(row["movie_review"]).strip()
        if not review_text or review_text.lower() == "nan":
            continue

        docs.append(
            Document(
                page_content=review_text,
                metadata={
                    "movie_name": str(row["movie_name"]),
                    "movie_rating": float(row["movie_rating"]) if str(row["movie_rating"]).lower() != "nan" else None,
                    "review_sentiment": str(row["review_sentiment"]),
                    "movie_url": str(row["movie_url"]),
                },
            )
        )

    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=REVIEWS_CHROMA_PATH,
    )

    try:
        vector_db.persist()
    except Exception:
        pass

    return vector_db

review_vector_db = build_or_load_reviews_db()
review_retriever = review_vector_db.as_retriever(search_kwargs={"k": 10})

chat_model = ChatOllama(
    model=consts.OLLAMA_MODEL,
    base_url=consts.OLLAMA_BASE_URL,
    temperature=0
)

review_chain = (
    {
        "question": itemgetter("question"),
        "context": itemgetter("question") | review_retriever | format_reviews,
    }
    | review_prompt_template
    | chat_model
    | StrOutputParser()
)

if __name__ == "__main__":
    questions = [
        "Summarize what people liked and disliked about the movie Stanley & Iris.",
        "Do reviews mention that the movie is slow-paced or boring?",
        "What do people say about the acting performances?",
    ]

    for q in questions:
        print("\nQUESTION:", q)
        print(review_chain.invoke({"question": q}))
