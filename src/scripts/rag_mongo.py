import sys
import os

from mongoengine import connect
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.abspath(os.path.join(current_dir, os.pardir)))

sys.path.append(parent_dir)

load_dotenv()

from src.utils.embeddings import generate_embeddings
from src.models.rag import RAGSegment
from src.utils.utils_csv import read_csv, line_to_str



import time





db = os.getenv('CONNECTA_DB_NAME')
host = os.getenv('CONNECTA_MONGO_URI')
connect(db=db, host=host)

def embeddings(connecta, bolsistas):
    segment_connecta = RAGSegment(
        context="connecta ceia",
        text = connecta,
        text_embedding="fale sobre conecta ceia",
        source_type="csv",
        embedding=generate_embeddings("fale sobre bolsistas conecta ceia"),
        instructions="Isso é um csv falando sobre o Centro de Excellencia em Inteligencia Artificial(CEIA)"
    )

    segment_bolsistas = RAGSegment(
        context="connecta ceia",
        text = bolsistas,
        text_embedding="fale sobre bolsistas conecta ceia",
        source_type="csv",
        embedding=generate_embeddings("fale sobre bolsistas conecta ceia"),
        instructions="isso é um csv falando sobre bolsistas do ceia, considerando suas posições dentro dos projetos que estão sendo desenvolvidos e o valor que recebem para desenvolver esses projetos"
    )

    RAGSegment.drop_collection()

    print("Aguarde para que o index possa ser criado novamente...")
    for i in range(30):
        print(i)
        time.sleep(1)

    segment_bolsistas.save()
    segment_connecta.save()
    RAGSegment.create_vector_index()

