from src.cypher_requests import (
    get_sentiment_analysis_of_actor,
    get_average_rating_for_movie,
    get_most_recent_movie,
    get_movies_not_reviewed_by_user,
    get_coactors_grouped_by_movie,
)

if __name__ == "__main__":
    actor_name = "Robert de Niro"
    movie_title = "Sounder"
    user_name = "Robert Brooks"

    get_sentiment_analysis_of_actor(actor_name)
    get_average_rating_for_movie(movie_title)
    get_most_recent_movie()
    get_movies_not_reviewed_by_user(user_name)
    get_coactors_grouped_by_movie(actor_name)