RAG_Growth : from transformer to Graph_RAG

# Objective

This project aims at exploring the use of transformers locally, from a simple LLM call up to a Graph_RAG, with increasingly complex steps.

# Composition

Each folder will contain the code necessary to implement one step of the project, with the code needed to collect the data if necessary. 
Despite 

# 0 - Local transformer

Implementation of a transformer deploy locally through a Jupyter notebook. 

Base model used: "mistralai/Ministral-3-3B-Instruct-2512"

# 1 - Tool Addition

Addition of the usage of tools to our base model

tool:
fetch_book_url (from the Gutenberg project)

# 2 - Agentic behavior

Modification of the structure 