import os

from langchain_classic.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from py2neo import Graph
import json

import src.user_consts as consts
from src.langchain.utils.neo4j_utils_py2neo import extract_cypher, correct_cypher_query

# Connexion à Neo4j
graph = Graph(consts.NEO4J_URI, auth=(consts.NEO4J_USERNAME, consts.NEO4J_PASSWORD))

# Configuration du modèle de LLM
agent_chat_model = ChatOllama(model=consts.OLLAMA_MODEL, base_url=consts.OLLAMA_BASE_URL, temperature=0)

# Charger le schéma de la base de données
SCHEMA_PATH = os.path.join(consts.IN_FOLDER, 'movie', 'neo4j_graph_schema.json')
with open(SCHEMA_PATH, 'r') as f:
    schema = json.load(f)

# Exemples de requêtes Cypher correctes
CYPHER_EXAMPLES = """
EXAMPLES OF CORRECT CYPHER QUERIES FROM YOUR DATABASE:

1. Sentiment analysis for an actor:
Question: "What is the sentiment for actor Tom Hanks?"
MATCH (a:Actor {{name: "Tom Hanks"}})-[:ACTED_IN]->(m:Movie)<-[:RATED]-(u:User)
RETURN CASE
    WHEN AVG(toFloat(u.rating)) >= 7 THEN "Positive"
    WHEN AVG(toFloat(u.rating)) >= 4 THEN "Neutral"
    ELSE "Negative"
END AS sentiment

2. Average rating for a movie:
Question: "What is the average rating for Sounder?"
MATCH (m:Movie {{title: "Sounder"}})<-[:RATED]-(u:User)
RETURN AVG(toFloat(u.rating)) AS average_rating

3. Most recent movie:
Question: "What is the most recent movie?"
MATCH (m:Movie)
RETURN m.title AS most_recent_movie
ORDER BY toInteger(m.released) DESC
LIMIT 1

4. Movies not reviewed by a user:
Question: "Which movies has Alice not reviewed?"
MATCH (m:Movie)
WHERE NOT EXISTS {{
    MATCH (u:User {{name: "Alice"}})-[:RATED]->(m)
}}
RETURN m.title AS movies_not_reviewed

5. Co-actors grouped by movie:
Question: "Who are the co-actors of Tom Hanks in each movie?"
MATCH (a:Actor {{name: "Tom Hanks"}})-[:ACTED_IN]->(m:Movie)<-[:ACTED_IN]-(co:Actor)
WHERE co.name <> "Tom Hanks"
RETURN m.title AS movie, COLLECT(co.name) AS co_actors
"""

# Template pour la génération de requêtes Cypher
CYPHER_GENERATION_TEMPLATE = """
You are an expert in Neo4j and Cypher. Your task is to translate a natural language query into a valid Cypher query.

Database schema:
{schema}

IMPORTANT CYPHER RULES:
- The 'released' property is a STRING (format 'YYYY'), not a DATE
- To find the maximum/minimum, use ORDER BY and LIMIT, NOT max() or min() in MATCH
- To search for text, use CONTAINS (e.g. WHERE m.title CONTAINS 'Matrix')
- Always use ORDER BY before LIMIT
- Aggregate functions (avg, count, max, min) are used in RETURN, not in MATCH or WHERE
- Use toFloat() or toInteger() for type conversions when needed

""" + CYPHER_EXAMPLES + """

Instructions:
- Generate ONLY the Cypher query, WITHOUT explanatory text
- Follow the pattern of the examples above
- Limit results with LIMIT if not specified (default 10)
- Sort results in a relevant manner
- Include only the necessary columns in RETURN

Question: {question}

Cypher query:
"""

# Template pour la réponse basée sur les résultats
QA_GENERATION_TEMPLATE = """
You are an assistant presenting the results of a film database in a clear and structured manner.

Question asked: {question}

Cypher query executed:
{query}

Database results:
{context}

Instructions:
- Present the results in a natural and readable way.
- Use bulleted or numbered lists where appropriate.
- Include relevant details (ratings, years, names).
- If there are no results, state this clearly.
- Respond in French but don't translate the title of the movie.

Answer:
"""

cypher_prompt = PromptTemplate(
    template=CYPHER_GENERATION_TEMPLATE,
    input_variables=["question", "schema"]
)

# Chaîne Cypher : transforme la question en requête Cypher
cypher_chain = LLMChain(
    llm=agent_chat_model,
    prompt=cypher_prompt,
    verbose=True
)

qa_prompt = PromptTemplate(
    template=QA_GENERATION_TEMPLATE,
    input_variables=["question", "query", "context"]
)

# Chaîne QA : génère une réponse textuelle à partir des résultats
qa_chain = LLMChain(
    llm=agent_chat_model,
    prompt=qa_prompt,
    verbose=True
)

def explore_movies(question: str):
    """
    Explore la base Neo4j en traduisant une question en Cypher et en générant une réponse.

    Args:
        question: Question en langage naturel sur les films/castings

    Returns:
        dict contenant la réponse, la requête Cypher et les résultats bruts
    """
    try:
        # Étape 1 : Générer la requête Cypher avec cypher_chain
        print(f"\n{'=' * 80}")
        print(f"Question : {question}")
        print(f"\nÉtape 1 : Génération de la requête Cypher...")

        cypher_result = cypher_chain.invoke({
            "question": question,
            "schema": json.dumps(schema, indent=2)
        })

        # Nettoyer la requête (enlever les backticks markdown si présents)
        cypher_query_text = cypher_result["text"].strip()
        extracted_cypher_query = extract_cypher(cypher_query_text)
        cypher_query = correct_cypher_query(extracted_cypher_query)

        print(f"\nRequête Cypher générée :")
        print(cypher_query)

        # Étape 2 : Exécuter la requête dans Neo4j
        print(f"\nÉtape 2 : Exécution de la requête dans Neo4j...")
        results = graph.run(cypher_query).data()

        print(f"\nNombre de résultats : {len(results)}")
        print(f"Résultats bruts :")
        print(results)

        # Étape 3 : Générer une réponse textuelle avec qa_chain
        print(f"\nÉtape 3 : Génération de la réponse textuelle...")
        qa_result = qa_chain.invoke({
            "question": question,
            "query": cypher_query,
            "context": json.dumps(results, indent=2, ensure_ascii=False)
        })

        answer = qa_result["text"]

        print(f"\nRéponse finale :")
        print(answer)
        print(f"{'=' * 80}\n")

        return {
            "question": question,
            "cypher_query": cypher_query,
            "database_results": results,
            "answer": answer
        }

    except Exception as e:
        print(f"Erreur lors de l'exploration : {e}")
        import traceback
        traceback.print_exc()
        return None

# Exemples d'utilisation
if __name__ == "__main__":
    question = "What is the most recent movie?"
    explore_movies(question)
