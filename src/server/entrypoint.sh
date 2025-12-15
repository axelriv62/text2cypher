#!/bin/bash

echo "Starting movies chatbot server..."

export PYTHONPATH="$PWD"/../../
fastapi run launch_fastapi_server.py