from py2neo import Graph

graph = Graph("bolt://localhost:7687", auth=("neo4j", ""))

def get_sentiment_analysis_of_actor(actor_name):
    query = """
    MATCH (a:Actor {name: $actor_name})-[:ACTED_IN]->(m:Movie)<-[:RATED]-(u:User)
    RETURN CASE
        WHEN AVG(toFloat(u.rating)) >= 7 THEN "Positive"
        WHEN AVG(toFloat(u.rating)) >= 4 THEN "Neutral"
        ELSE "Negative"
    END AS sentiment
    """
    result = graph.run(query, actor_name=actor_name).data()
    print(f"Sentiment analysis for actor '{actor_name}': {result}")
    return result

def get_average_rating_for_movie(movie_title):
    query = """
    MATCH (m:Movie {title: $movie_title})<-[:RATED]-(u:User)
    RETURN AVG(toFloat(u.rating)) AS average_rating
    """
    result = graph.run(query, movie_title=movie_title).data()
    print(f"Average rating for movie '{movie_title}': {result}")
    return result

def get_most_recent_movie():
    query = """
    MATCH (m:Movie)
    RETURN m.title AS most_recent_movie
    ORDER BY toInteger(m.released) DESC
    LIMIT 1
    """
    result = graph.run(query).data()
    print(f"Most recent movie: {result}")
    return result

def get_movies_not_reviewed_by_user(user_name):
    query = """
    MATCH (m:Movie)
    WHERE NOT EXISTS {
        MATCH (u:User {name: $user_name})-[:RATED]->(m)
    }
    RETURN m.title AS movies_not_reviewed
    """
    result = graph.run(query, user_name=user_name).data()
    print(f"Movies not reviewed by user '{user_name}': {result}")
    return result

def get_coactors_grouped_by_movie(actor_name):
    query = """
    MATCH (a:Actor {name: $actor_name})-[:ACTED_IN]->(m:Movie)<-[:ACTED_IN]-(co:Actor)
    WHERE co.name <> $actor_name
    RETURN m.title AS movie, COLLECT(co.name) AS co_actors
    """
    result = graph.run(query, actor_name=actor_name).data()
    print(f"Co-actors grouped by movie for actor '{actor_name}': {result}")
    return result

