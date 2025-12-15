from py2neo import Graph

graph = Graph("bolt://localhost:7687", auth=("neo4j", ""))

# 1. What is the overall sentiment analysis regarding the acting of Robert de Niro ? Keep it as one word.
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

# 2. What is the average rating value for the movie Sounder ?
def get_average_rating_for_movie(movie_title):
    query = """
    MATCH (m:Movie {title: $movie_title})<-[:RATED]-(u:User)
    RETURN AVG(toFloat(u.rating)) AS average_rating
    """
    result = graph.run(query, movie_title=movie_title).data()
    print(f"Average rating for movie '{movie_title}': {result}")
    return result

# 3. What is the most recent movie ?
def get_most_recent_movie():
    query = """
    MATCH (m:Movie)
    RETURN m.title, m.released
    ORDER BY m.released DESC
    LIMIT 1
    """
    result = graph.run(query).data()
    print(f"Most recent movie: {result}")
    return result

# 4. Which movies are not reviewed by Robert Brooks ?
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

# 5. What are the co-actors of Robert de Niro, grouped by movie ?
def get_coactors_grouped_by_movie(actor_name):
    query = """
    MATCH (a:Actor {name: $actor_name})-[:ACTED_IN]->(m:Movie)<-[:ACTED_IN]-(co:Actor)
    WHERE co.name <> $actor_name
    RETURN m.title AS movie, COLLECT(co.name) AS co_actors
    """
    result = graph.run(query, actor_name=actor_name).data()
    print(f"Co-actors grouped by movie for actor '{actor_name}': {result}")
    return result

# 6. List the directors who directed more than one movie.
def get_directors_with_multiple_movies():
    query = """
    MATCH (d:Director)-[:DIRECTED]->(m:Movie)
    WITH d, COUNT(m) AS movie_count
    WHERE movie_count > 1
    RETURN d.name AS director, movie_count
    """
    result = graph.run(query).data()
    print(f"Directors who directed more than one movie: {result}")
    return result

# 7. List the American directors and actors
def get_american_directors_and_actors():
    query = """
    MATCH (p:Person)
    WHERE p.nationality = 'American' AND (p:Director OR p:Actor)
    RETURN p.name AS name, labels(p) AS roles
    """
    result = graph.run(query).data()
    print(f"American directors and actors: {result}")
    return result

# 8. List the directors older than 80 years old
def get_directors_older_than(age):
    query = """
    MATCH (d:Director)
    WHERE d.age > $age
    RETURN d.name AS director, d.age AS age
    """
    result = graph.run(query, age=age).data()
    print(f"Directors older than {age} years old: {result}")
    return result

# 9. List the died directors
def get_died_directors():
    query = """
    MATCH (d:Director)
    WHERE EXISTS(d.deathDate)
    RETURN d.name AS director, d.deathDate AS death_date
    """
    result = graph.run(query).data()
    print(f"Died directors: {result}")
    return result

# 10. Query the database to list romantic movies
def get_romantic_movies():
    query = """
    MATCH (m:Movie)-[:IN_GENRE]->(g:Genre {name: 'Romance'})
    RETURN m.title AS romantic_movies
    """
    result = graph.run(query).data()
    print(f"Romantic movies: {result}")
    return result