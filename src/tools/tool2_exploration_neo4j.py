from langchain_classic.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from py2neo import Graph
import json

import src.user_consts as consts

# Connexion à Neo4j
graph = Graph(consts.NEO4J_URI, auth=(consts.NEO4J_USERNAME, consts.NEO4J_PASSWORD))

# Configuration du modèle de LLM
agent_chat_model = ChatOllama(model=consts.OLLAMA_MODEL, base_url=consts.OLLAMA_BASE_URL, temperature=0)

# Charger le schéma de la base de données
with open('../../in/movie/neo4j_graph_schema.json', 'r') as f:
    schema = json.load(f)

# Template pour la génération de requêtes Cypher
CYPHER_GENERATION_TEMPLATE = """
You are an expert in Neo4j and Cypher. Your task is to translate a natural language query into a valid Cypher query.

Database schema:
{schema}

IMPORTANT CYPHER RULES:
- The ‘released’ property is a STRING (format ‘YYYY’), not a DATE
- To find the maximum/minimum, use ORDER BY and LIMIT, NOT max() or min() in MATCH
- To search for text, use CONTAINS (e.g. WHERE m.title CONTAINS ‘Matrix’)
- Always use ORDER BY before LIMIT
- Aggregate functions (avg, count, max, min) are used in RETURN, not in MATCH or WHERE

EXAMPLES OF CORRECT QUERIES:
1. Most recent movie:
MATCH (m:Movie)
WHERE m.released IS NOT NULL
RETURN m.title, m.released
ORDER BY m.released DESC
LIMIT 1

2. Actors in a film:
MATCH (a:Actor)-[r:ACTED_IN]->(m:Movie)
WHERE m.title CONTAINS ‘Matrix’
RETURN a.name, r.role

3. Average rating of a movie:
MATCH (u:User)-[r:RATED]->(m:Movie)
WHERE m.title = ‘Sounder’
RETURN m.title, avg(r.rating) as average_rating

Instructions:
- Generate ONLY the Cypher query, WITHOUT explanatory text
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
        cypher_query = clean_cypher_query(cypher_result["text"].strip())

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


def clean_cypher_query(cypher_text: str) -> str:
    """
    Nettoie la question en supprimant les caractères spéciaux inutiles.

    Args:
        question: Question en langage naturel

    Returns:
        Question nettoyée
    """
    # Nettoyage avancé de la requête
    # 1. Enlever les backticks markdown
    cypher_text = cypher_text.replace("```cypher", "").replace("```", "").strip()

    # 2. Extraire uniquement la partie avant toute explication
    lines = cypher_text.split('\n')
    cypher_lines = []

    for line in lines:
        # Arrêter si on trouve une explication (commence par mot en minuscule ou phrase)
        if line.strip() and not line.strip().startswith(('MATCH', 'WITH', 'WHERE', 'RETURN',
                                                         'ORDER', 'LIMIT', 'OPTIONAL',
                                                         'CREATE', 'MERGE', 'DELETE',
                                                         'SET', 'REMOVE', 'UNWIND',
                                                         '//', '--')):
            # Si la ligne ne commence pas par un mot-clé Cypher ou commentaire, c'est probablement une explication
            if cypher_lines:  # Si on a déjà des lignes Cypher, on arrête
                break
            else:  # Sinon on saute cette ligne
                continue
        cypher_lines.append(line)

    cypher_query = '\n'.join(cypher_lines).strip()

    # 3. Enlever les commentaires qui pourraient rester
    cypher_query = '\n'.join([
        line for line in cypher_query.split('\n')
        if not line.strip().startswith(('//', '--'))
    ]).strip()

    return cypher_query

# Exemples d'utilisation
if __name__ == "__main__":
    questions = [
        "Which movies are not reviewed by Robert Brooks?",
    ]

#         "What is the most recent movie?",
#         "What is the average rating value for the movie Sounder?",
    #         "What is the overall sentiment analysis regarding the acting of Robert de Niro? Keep it as one word?",
    #         "Which movies are not reviewed by Robert Brooks?",
    #         "What are the co-actors of Robert de Niro, grouped by movie?"

    for q in questions[:2]:
        explore_movies(q)

# todo