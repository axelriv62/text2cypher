from py2neo import Graph
import src.user_consts as consts

graph = Graph(
    consts.NEO4J_URI,
    auth=(consts.NEO4J_USERNAME, consts.NEO4J_PASSWORD)
)


def get_similarity(concept: str) -> float:
    name1, name2 = [x.strip() for x in concept.split(",")]

    query = """
    MATCH (u1:User {name: $n1}), (u2:User {name: $n2})
    OPTIONAL MATCH (u1)--(a)
    OPTIONAL MATCH (u2)--(b)
    WITH collect(DISTINCT id(a)) AS n1n, collect(DISTINCT id(b)) AS n2n
    UNWIND (n1n + n2n) AS allId
    WITH collect(DISTINCT allId) AS unionList, n1n, n2n
    WITH size([x IN n1n WHERE x IN n2n]) AS inter, size(unionList) AS unionSize
    RETURN CASE WHEN unionSize = 0 THEN 0.0 ELSE toFloat(inter) / toFloat(unionSize) END AS similarity
    """

    result = graph.run(query, n1=name1, n2=name2).data()

    if result:
        result = result[0]["similarity"]
    else:
        result = 0.0

    return result

#print(get_similarity("Sean Mendez,David Malone"))