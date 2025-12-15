#!/bin/bash

echo "Starting movie chatbot frontend..."

export PYTHONPATH="$PWD"/../../
streamlit run launch_streamlit_client.py
