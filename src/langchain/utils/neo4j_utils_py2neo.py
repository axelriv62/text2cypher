from src.langchain.utils.cypher_utils import CypherQueryCorrector, Schema
from typing import Any, Dict, List
import re
import src.user_consts as consts
from py2neo import Graph
import json
import os



def construct_neo4j_graph_schema(include_types: List[str], exclude_types: List[str]) -> str:
    """Filter the schema based on included or excluded types"""
    #graph = Graph(consts.NEO4J_URI, auth=(consts.NEO4J_USERNAME, consts.NEO4J_PASSWORD))
    #graph.refresh_schema()
    return construct_schema(include_types, exclude_types)

def run_predicted_cypher_query(generated_cypher: str, top_k:int=10, node_properties_to_exclude=[]) -> list:
    context = []
    print(generated_cypher)
    graph = Graph(consts.NEO4J_URI, auth=(consts.NEO4J_USERNAME, consts.NEO4J_PASSWORD))
    #graph.refresh_schema()

    context = graph.run(generated_cypher).data()[:top_k]

    if node_properties_to_exclude and isinstance(context, list):
        context = remove_keys_from_dicts(
            context, node_properties_to_exclude
        )

    return context

def extract_cypher(text: str) -> str:
    """Extract Cypher code from a text.

    Args:
        text: Text to extract Cypher code from.

    Returns:
        Cypher code extracted from the text.
    """
    # The pattern to find Cypher code enclosed in triple backticks
    pattern = r"```(.*?)```"

    # Find all matches in the input text
    matches = re.findall(pattern, text, re.DOTALL)

    return matches[0] if matches else text

def construct_schema(
    include_types: List[str],
    exclude_types: List[str],
) -> str:
    """Filter the schema based on included or excluded types"""

    #graph = Graph(consts.NEO4J_URI, auth=(consts.NEO4J_USERNAME, consts.NEO4J_PASSWORD))
    #res = graph.run("call db.schema.visualization()").data()[0]
    #print(res["nodes"])
    #print(res["relationships"])
    with open(os.path.join(consts.IN_FOLDER, 'movie', 'neo4j_graph_schema.json'), "r") as f:
        structured_schema = json.load(f)

    def filter_func(x: str) -> bool:
        return x in include_types if include_types else x not in exclude_types

    filtered_schema: Dict[str, Any] = {
        "node_props": {
            k: v
            for k, v in structured_schema.get("node_props", {}).items()
            if filter_func(k)
        },
        "rel_props": {
            k: v
            for k, v in structured_schema.get("rel_props", {}).items()
            if filter_func(k)
        },
        "relationships": [
            r
            for r in structured_schema.get("relationships", [])
            if all(filter_func(r[t]) for t in ["start", "end", "type"])
        ],
    }

    # Format node properties
    formatted_node_props = []
    for label, properties in filtered_schema["node_props"].items():
        props_str = ", ".join(
            [f"{prop['property']}: {prop['type']}" for prop in properties]
        )
        formatted_node_props.append(f"{label} {{{props_str}}}")

    # Format relationship properties
    formatted_rel_props = []
    for rel_type, properties in filtered_schema["rel_props"].items():
        props_str = ", ".join(
            [f"{prop['property']}: {prop['type']}" for prop in properties]
        )
        formatted_rel_props.append(f"{rel_type} {{{props_str}}}")

    # Format relationships
    formatted_rels = [
        f"(:{el['start']})-[:{el['type']}]->(:{el['end']})"
        for el in filtered_schema["relationships"]
    ]

    return "\n".join(
        [
            "Node properties are the following:",
            ",".join(formatted_node_props),
            "Relationship properties are the following:",
            ",".join(formatted_rel_props),
            "The relationships are the following:",
            ",".join(formatted_rels),
        ]
    )

def correct_cypher_query(generated_cypher: str) -> str:
    #graph = Neo4jGraph(
    #    url=consts.NEO4J_URI,
    #    username=consts.NEO4J_USERNAME,
    #    password=consts.NEO4J_PASSWORD,
    #)
    #graph.refresh_schema()
    with open(os.path.join(consts.IN_FOLDER, 'movie', 'neo4j_graph_schema.json'), "r") as f:
        structured_schema = json.load(f)

    corrector_schema = [
        Schema(el["start"], el["type"], el["end"])
        for el in structured_schema.get("relationships")
    ]
    cypher_query_corrector = CypherQueryCorrector(corrector_schema)

    # Correct Cypher query if enabled
    if cypher_query_corrector:
        corrected_cypher = cypher_query_corrector(generated_cypher)
        if "→" in corrected_cypher:
            corrected_cypher = corrected_cypher.replace("→", "->")
            corrected_cypher = corrected_cypher.replace("←", "<-")
        return corrected_cypher
    return "-1"

def remove_keys_from_dicts(input_list: list, keys_to_remove: list):
    """Remove keys from a nested dictionary"""

    def remove_keys_from_dict(d, keys):
        if isinstance(d, dict):
            d = {
                k: (remove_keys_from_dict(v, keys) if isinstance(v, dict) else v)
                for k, v in d.items()
                if k not in keys
            }
        elif isinstance(d, list):
            d = [remove_keys_from_dict(i, keys) for i in d]
        return d

    return [remove_keys_from_dict(item, keys_to_remove) for item in input_list]
