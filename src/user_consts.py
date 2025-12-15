import os

PROJECT_ROOT = "/home/axel/Projects/bdd/projet-text2cypher"
IN_FOLDER = os.path.join(PROJECT_ROOT, 'in')
OUT_FOLDER = os.path.join(PROJECT_ROOT, 'out')

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "mot2passe"

OLLAMA_MODEL = "llama3.2:1b"
OLLAMA_BASE_URL = "http://localhost:11434" # Default Ollama URL
OLLAMA_TEXT_SIMILARITY_MODEL = "chroma/all-minilm-l6-v2-f32"
