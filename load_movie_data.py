from py2neo import Graph, Node, Relationship
import pandas as pd
import src.user_consts as consts
import os
import logging

# Configure the logging module
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


def delete_all_data_from_neo4j():
    graph = Graph(consts.NEO4J_URI, auth=(consts.NEO4J_USERNAME, consts.NEO4J_PASSWORD))
    graph.delete_all()
    # cypher query:
    # MATCH (n) DETACH DELETE n

def load_movie_graph_from_csv_for_neo4j():
    graph = Graph(consts.NEO4J_URI, auth=(consts.NEO4J_USERNAME, consts.NEO4J_PASSWORD))
    graph.delete_all()

    movie_ids = [3103, 3826, 6268, 1177, 41716, 41226, 1199]

    in_folder = consts.IN_FOLDER

    ######################################################
    # Loading 'Movie' nodes"
    ######################################################
    LOGGER.info("------   Loading 'Movie' nodes")
    df_movie = pd.read_csv(os.path.join(in_folder, "movie","movie.csv"), sep=',', dtype=str)
    for index, row in df_movie.iterrows():
        if int(row['movieId']) in movie_ids:
            movie_node = Node("Movie", movieId=int(row['movieId']), imdbId=str(row["imdbId"]), title=row['title'], imdbRating=row['imdbRating'],
                              url=row['url'], revenue=row['revenue'], tmdbId=int(row['tmdbId']), plot=row['plot'],
                              poster=row['poster'], released=row['released'], budget=row['budget'])
            graph.create(movie_node)


    ######################################################
    # Loading 'Genre' nodes"
    ######################################################
    LOGGER.info("------   Loading 'Genre' nodes and their relationships")
    df_genre = pd.read_csv(os.path.join(in_folder, "movie", "Genre.csv"), sep='\t')
    df_in_genre = pd.read_csv(os.path.join(in_folder, "movie","in_genre.csv"), sep=',')
    for index, row in df_in_genre.iterrows():
        movie_id = int(row['movieId'])
        if movie_id in movie_ids:
            movie_node = graph.nodes.match("Movie", movieId=movie_id).first()
            genre_name = row['genre']
            genre_node = graph.nodes.match("Genre", name=genre_name).first()
            if genre_node is None:
                genre_row = df_genre[df_genre['name'] == genre_name].iloc[0]
                genre_node = Node("Genre", name=genre_row['name'])
                graph.create(genre_node)
            if movie_node is not None:
                relationship = Relationship(movie_node, "IN_GENRE", genre_node)
                graph.create(relationship)


    ######################################################
    # Loading 'Actor' nodes"
    ######################################################
    LOGGER.info("------   Loading 'Actor' nodes and their relationships")
    df_actor = pd.read_csv(os.path.join(in_folder, "movie","Actor.csv"), sep=',')
    df_acted_in = pd.read_csv(os.path.join(in_folder, "movie","acted_in.csv"), sep=',')
    for index, row in df_acted_in.iterrows():
        movie_id = int(row['movieId'])
        if movie_id in movie_ids:
            movie_node = graph.nodes.match("Movie", movieId=movie_id).first()
            actor_name = row['name']
            actor_row = df_actor[df_actor['name'] == actor_name].iloc[0]
            actor_node = graph.nodes.match("Actor", imdbId=str(actor_row['imdbId'])).first()
            actor_role = row['actor_role']
            if actor_node is None:
                actor_node = Node("Actor", bornIn=actor_row['bornIn'], imdbId=str(actor_row['imdbId']), tmdbId=int(actor_row['tmdbId']), name=actor_row['name'],
                                  born=actor_row['born'], died=actor_row['died'], url=actor_row['url'])
                graph.create(actor_node)
            if movie_node is not None:
                relationship = Relationship(actor_node, "ACTED_IN", movie_node, role=actor_role)
                graph.create(relationship)


    ######################################################
    # Loading 'Director' nodes"
    ######################################################
    LOGGER.info("------   Loading 'Director' nodes and their relationships")
    df_director = pd.read_csv(os.path.join(in_folder, "movie","Director.csv"), sep=',', na_filter=False, dtype = {'died': str})
    df_directed = pd.read_csv(os.path.join(in_folder, "movie","directed.csv"), sep=',')
    for index, row in df_directed.iterrows():
        movie_id = int(row['movieId'])
        if movie_id in movie_ids:
            movie_node = graph.nodes.match("Movie", movieId=movie_id).first()
            director_name = row['name']
            director_row = df_director[df_director['name'] == director_name].iloc[0]
            director_node = graph.nodes.match("Director", imdbId=str(director_row['tmdbId'])).first()
            if director_node is None:
                director_node = Node("Director", bornIn=director_row['bornIn'], imdbId=str(director_row['tmdbId']), born=director_row['born'], name=director_row['name'],
                                     bio=director_row['bio'], died=director_row['died'], poster=director_row['poster'], url=director_row['url'])
                graph.create(director_node)
            if movie_node is not None:
                relationship = Relationship(director_node, "DIRECTED", movie_node)
                graph.create(relationship)


    ######################################################
    # Loading 'User' nodes"
    ######################################################
    LOGGER.info("------   Loading 'User' nodes and their relationships")
    df_User = pd.read_csv(os.path.join(in_folder, "movie","User.csv"), sep=',')
    df_rated = pd.read_csv(os.path.join(in_folder, "movie","rated.csv"), sep=',')
    for index, row in df_rated.iterrows():
        movie_id = int(row['movieId'])
        if movie_id in movie_ids:
            movie_node = graph.nodes.match("Movie", movieId=movie_id).first()
            user_name = row['name']
            user_row = df_User[df_User['name'] == user_name].iloc[0]
            userId = int(user_row['userId'])
            user_node = graph.nodes.match("User", userId=userId).first()
            rating = row['rating']
            if user_node is None:
                user_node = Node("User", userId=int(user_row['userId']), name=user_row['name'])
                graph.create(user_node)
            if movie_node is not None:
                relationship = Relationship(user_node, "RATED", movie_node, rating=rating)
                graph.create(relationship)



if __name__ == '__main__':
    LOGGER.info("Reset all existing data in the Neo4j database ...")
    delete_all_data_from_neo4j()

    LOGGER.info("Started to load the movie data ...")
    load_movie_graph_from_csv_for_neo4j()
    LOGGER.info("The movie data has been successfully loaded !!")