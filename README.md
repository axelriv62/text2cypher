# Projet BDD - Requêtage de base de données Neo4j à partir de texte en langage naturel via des LLM
Par Axel Rivière, Romain Notteau et Quentin Tripognez (BUT3 APP)

## Rapport
Vous pouvez trouver le rapport du projet [ici](docs/rapport-riviere-notteau-tripognez.pdf).

## Tests
Vous pouvez trouver la trace des tests [ici](docs/tests-trace.txt).

## Lancer le projet
```bash
# Installer les dépendances
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Lancer le serveur
cd src/server 
chmod u+x entrypoint.sh
./entrypoint.sh

# Lancer le client
cd src/client 
chmod u+x entrypoint.sh
./entrypoint.sh
```
