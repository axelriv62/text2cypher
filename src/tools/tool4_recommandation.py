from py2neo import Graph
import src.user_consts as consts

# Connexion à la base de données
graph = Graph(consts.NEO4J_URI, auth=(consts.NEO4J_USERNAME, consts.NEO4J_PASSWORD))

def get_recommendation(concept_name: str):
    """
    Recommande des films basés sur un concept (genre, acteur ou réalisateur)

    Args:
        concept_name: Le nom du concept (ex: "Drama", "Donna Maldonado", "Peter Sollett")

    Returns:
        Liste de films recommandés avec leurs détails
    """
    query = """
        // Rechercher le concept dans différentes catégories
        OPTIONAL MATCH (g:Genre {name: $concept})
        OPTIONAL MATCH (a:Actor) WHERE a.name CONTAINS $concept
        OPTIONAL MATCH (d:Director) WHERE d.name CONTAINS $concept

        WITH g, a, d

        // Récupérer les films correspondants
        OPTIONAL MATCH (m1:Movie)-[:IN_GENRE]->(g)
        OPTIONAL MATCH (a)-[:ACTED_IN]->(m2:Movie)
        OPTIONAL MATCH (d)-[:DIRECTED]->(m3:Movie)

        WITH collect(DISTINCT m1) + collect(DISTINCT m2) + collect(DISTINCT m3) AS movies
        UNWIND movies AS movie

        WITH movie
        WHERE movie IS NOT NULL

        // Enrichir avec les informations du film
        MATCH (movie)-[:IN_GENRE]->(genre:Genre)
        OPTIONAL MATCH (actor:Actor)-[act:ACTED_IN]->(movie)
        OPTIONAL MATCH (director:Director)-[:DIRECTED]->(movie)
        OPTIONAL MATCH (u:User)-[r:RATED]->(movie)

        WITH movie,
             collect(DISTINCT genre.name) AS genres,
             collect(DISTINCT {name: actor.name, role: act.role}) AS actors,
             collect(DISTINCT director.name) AS directors,
             avg(r.rating) AS avg_rating,
             count(DISTINCT r) AS rating_count

        RETURN movie.title AS title,
               movie.released AS released,
               movie.plot AS plot,
               movie.imdbRating AS imdbRating,
               movie.poster AS poster,
               genres,
               actors AS top_actors,
               directors,
               avg_rating,
               rating_count
        ORDER BY avg_rating DESC, rating_count DESC, movie.imdbRating DESC
        LIMIT 10
    """

    res = graph.run(query, concept=concept_name).data()

    if res:
        print(f"\nRecommandations de films pour '{concept_name}':\n")
        for i, movie in enumerate(res, 1):
            print(f"{i}. {movie['title']} ({movie['released']})")
            print(f"   - Note IMDB: {movie['imdbRating']}")
            if movie['avg_rating']:
                print(f"   - Note moyenne utilisateurs: {movie['avg_rating']:.2f} ({movie['rating_count']} avis)")
            print(f"   - Genres: {', '.join(movie['genres'])}")
            if movie['directors']:
                print(f"   - Réalisateur(s): {', '.join(movie['directors'])}")
            if movie['top_actors']:
                actors_str = ', '.join([f"{a['name']} ({a['role']})" for a in movie['top_actors'] if a['name']])
                if actors_str:
                    print(f"   - Acteurs principaux: {actors_str}")
            if movie['plot']:
                print(f"   - Synopsis: {movie['plot']}")
            if i < len(res):
                print()
    else:
        print(f"Aucun film trouvé pour le concept '{concept_name}'.")

    return res

if __name__ == "__main__":
    concept = "Peter Sollett"
    get_recommendation(concept)