import os

from mongoengine import connect

from src.utils.embeddings import generate_embeddings
from src.models.rag import RAGSegment
from src.utils.utils_csv import read_csv, line_to_str
from dotenv import load_dotenv

import time

load_dotenv('../.env')

db = os.getenv('CONNECTA_DB_NAME')
host = os.getenv('CONNECTA_MONGO_URI')
connect(db=db, host=host)

segment = RAGSegment(
    context="connecta ceia",
    text = "\n".join([line_to_str(line) for line in read_csv('../data/bolsistas.csv')]),
    text_embedding="fale sobre bolsistas conecta ceia",
    source_type="csv",
    embedding=generate_embeddings("fale sobre bolsistas conecta ceia")
)

RAGSegment.drop_collection()

print("Aguarde para que o index possa ser criado novamente...")
for i in range(30):
    print(i)
    time.sleep(1)

segment.save()
RAGSegment.create_vector_index()

