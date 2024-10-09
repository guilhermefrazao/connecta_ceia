from src.models.rag import RAGSegment
from src.utils.embeddings import generate_embeddings
from src.utils.utils_csv import tokenize_text
from pymongo import MongoClient

import os
import pandas as pd

def mongodb_vector_search(vector_index, text, context, k):

    query_vector = generate_embeddings(text)

    mongo_client = MongoClient(os.getenv("CONNECTA_MONGO_URI"))
    
    db = mongo_client["connecta_ceia"]

    collection = db["connecta_rag"]  

    df_data_bolsistas = pd.read_csv("src/data/projetos_equipes_formatado.csv")

    df_data_CEIA = pd.read_csv("src/data/conecta_ceia_info.csv")

    dict_data_CEIA = df_data_CEIA.to_dict(orient='records')

    dict_data_bolsistas = df_data_bolsistas.to_dict(orient='records')

    collection.delete_many({})

    connecta = collection.insert_many(dict_data_CEIA)

    bolsistas = collection.insert_many(dict_data_bolsistas)

    #Aqui está dando problemas de timeout devido a ausencia da collection
    num_candidates = RAGSegment.objects.count()

    context = tokenize_text(context)

    pipeline = [
        {
            '$vectorSearch': {
                'index':"vector_index",
                'path': 'embedding',
                'queryVector': query_vector,
                'numCandidates': num_candidates,
                'limit': k if k <= num_candidates else num_candidates,
                'filter':{
                    "context": context
                }
            } 
        },
        {
            '$project': {
                '_id': 0,
                'context': 1,
                'text': 1,
                'text_embedding':1,
                'Nome': 1,
                'Descrição': 1,
                'score': {
                    '$meta': 'vectorSearchScore'
                }

            }
        }
    ]
    print("pipeline", pipeline[0]['$vectorSearch']['limit'])
    result = RAGSegment.objects().aggregate(pipeline)
    print("result", list(result))
    return list(result)