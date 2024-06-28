import os

# define init index
INIT_INDEX = os.getenv('INIT_INDEX', 'false').lower() == 'true'

# vector index persist directory
INDEX_PERSIST_DIRECTORY = os.getenv('INDEX_PERSIST_DIRECTORY', "./data/faiss_db")

# vector index name
INDEX_NAME = os.getenv('INDEX_NAME', "code_assistant_index")

# target url to scrape
TARGET_URL =  os.getenv('TARGET_URL', "https://docs.ansible.com/ansible/latest")

# 