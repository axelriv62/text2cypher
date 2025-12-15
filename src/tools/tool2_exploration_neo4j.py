import os

from langchain_classic.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from py2neo import Graph
import json

import src.user_consts as consts
from src.langchain.utils.neo4j_utils_py2neo import extract_cypher, correct_cypher_query, run_predicted_cypher_query

# Connexion à Neo4j
graph = Graph(consts.NEO4J_URI, auth=(consts.NEO4J_USERNAME, consts.NEO4J_PASSWORD))

# Configuration du modèle de LLM
agent_chat_model = ChatOllama(model=consts.OLLAMA_MODEL, base_url=consts.OLLAMA_BASE_URL, temperature=0)

# Charger le schéma de la base de données
SCHEMA_PATH = os.path.join(consts.IN_FOLDER, 'movie', 'neo4j_graph_schema.json')
with open(SCHEMA_PATH, 'r') as f:
    schema = json.load(f)

# Exemples de requêtes Cypher correctes
# Exemples de requêtes Cypher correctes (accolades échappées)
CYPHER_EXAMPLES = """
EXAMPLES OF CORRECT CYPHER QUERIES FROM YOUR DATABASE:

1. Most recent movie
Question: "What is the most recent movie?"
MATCH (m:Movie)
RETURN m.title, m.released
ORDER BY m.released DESC
LIMIT 1

2. Count of movies
Question: "How many movies are in the database?"
MATCH (m:Movie)
RETURN COUNT(m) AS movie_count

3. Average rating for a movie
Question: "What is the average rating for Sounder?"
MATCH (m:Movie {{title: "Sounder"}})<-[:RATED]-(u:User)
RETURN AVG(toFloat(u.rating)) AS average_rating

4. Movies not reviewed by a user
Question: "Which movies has Alice not reviewed?"
MATCH (m:Movie)
WHERE NOT EXISTS {{
    MATCH (u:User {{name: "Alice"}})-[:RATED]->(m)
}}
RETURN m.title AS movies_not_reviewed

5. Co-actors grouped by movie
Question: "Who are the co-actors of Tom Hanks in each movie?"
MATCH (a:Actor {{name: "Tom Hanks"}})-[:ACTED_IN]->(m:Movie)<-[:ACTED_IN]-(co:Actor)
WHERE co.name <> "Tom Hanks"
RETURN m.title AS movie, COLLECT(co.name) AS co_actors

6. Sentiment analysis for an actor
Question: "What is the sentiment for actor Tom Hanks?"
MATCH (a:Actor {{name: "Tom Hanks"}})-[:ACTED_IN]->(m:Movie)<-[:RATED]-(u:User)
RETURN CASE
    WHEN AVG(toFloat(u.rating)) >= 7 THEN "Positive"
    WHEN AVG(toFloat(u.rating)) >= 4 THEN "Neutral"
    ELSE "Negative"
END AS sentiment

7. Top 3 actors by movie count
Question: "Who are the top 3 actors with the most movies?"
MATCH (a:Actor)-[:ACTED_IN]->(m:Movie)
RETURN a.name AS actor, COUNT(m) AS movie_count
ORDER BY movie_count DESC
LIMIT 3

8. Users who rated both movies
Question: "Which users rated both The Matrix and Inception?"
MATCH (u:User)-[:RATED]->(m1:Movie {{title: "The Matrix"}})
MATCH (u)-[:RATED]->(m2:Movie {{title: "Inception"}})
RETURN DISTINCT u.name AS user

9. Movies in a genre with high ratings
Question: "What are the highest-rated Drama movies?"
MATCH (m:Movie)-[:IN_GENRE]->(g:Genre {{name: "Drama"}})
MATCH (u:User)-[:RATED]->(m)
WITH m, AVG(toFloat(u.rating)) AS avg_rating
WHERE avg_rating >= 7
RETURN m.title AS movie, avg_rating
ORDER BY avg_rating DESC

10. Directors who worked with an actor
Question: "Which directors have worked with Tom Hanks?"
MATCH (a:Actor {{name: "Tom Hanks"}})-[:ACTED_IN]->(m:Movie)<-[:DIRECTED]-(d:Director)
RETURN DISTINCT d.name AS director
"""

# Template pour la génération de requêtes Cypher
CYPHER_GENERATION_TEMPLATE = """
You are an expert in Neo4j and Cypher. Your task is to translate a natural language query into a valid Cypher query.

Database schema:
{schema}

IMPORTANT CYPHER RULES:
- The 'released' property is a STRING (format 'YYYY'), not a DATE
- NEVER use MAX() or MIN() inside MATCH or WHERE clauses
- NEVER use MAX() or MIN() in property filters like {{released: MAX(...)}}
- To find maximum/minimum values: use ORDER BY with LIMIT
- To search for text, use CONTAINS (e.g. WHERE m.title CONTAINS 'Matrix')
- Always use ORDER BY before LIMIT
- Aggregate functions (avg, count) are ONLY used in RETURN or WITH clauses
- Use toFloat() or toInteger() for type conversions when needed

WRONG EXAMPLES (NEVER DO THIS):
❌ MATCH (m:Movie {{released: MAX(released)}})
❌ MATCH (m:Movie {{released: MIN(released)}})
❌ MATCH (m:Movie) WHERE m.released = MAX(m.released)
❌ MATCH (m:Movie {{released: (SELECT MAX(released) FROM Movie)}})

CORRECT PATTERN for finding maximum/minimum:
✅ MATCH (m:Movie) RETURN m ORDER BY m.released DESC LIMIT 1
✅ MATCH (m:Movie) RETURN m ORDER BY toInteger(m.released) DESC LIMIT 1

""" + CYPHER_EXAMPLES + """

Instructions:
- Generate ONLY the Cypher query, WITHOUT explanatory text or markdown
- Do NOT include the question number or "Question:" prefix
- Do NOT use MAX() or MIN() in MATCH clauses
- Follow the EXACT pattern of example 1 for "most recent" questions
- Limit results with LIMIT if not specified (default 10)
- Sort results in a relevant manner
- Include only the necessary columns in RETURN

Question: {question}

Cypher query (output ONLY the Cypher code):
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
        cypher_result = cypher_chain.invoke({
            "question": question,
            "schema": json.dumps(schema, indent=2)
        })

        # Nettoyer la requête (enlever les backticks markdown si présents)
        cypher_query_text = cypher_result["text"].strip()
        extracted_cypher_query = extract_cypher(cypher_query_text)
        cypher_query = correct_cypher_query(extracted_cypher_query)

        # Étape 2 : Exécuter la requête dans Neo4j
        print(f"\nÉtape 2 : Exécution de la requête dans Neo4j...")
        results = run_predicted_cypher_query(cypher_query)

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



######################
# TEST
######################
if __name__ == "__main__":
    context = "What is the most recent movie?"
    explore_movies(context)
