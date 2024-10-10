from src.models.rag import RAGSegment
from src.utils.embeddings import generate_embeddings
from src.utils.utils_csv import tokenize_text
from src.scripts.rag_mongo import embeddings
from pymongo import MongoClient

import os
import pandas as pd

def mongodb_vector_search(vector_index, text, context, k):

    query_vector = generate_embeddings(text)
    
    #Aqui está dando problemas de timeout devido a ausencia da collection
    num_candidates = RAGSegment.objects.count()


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

                'score': {
                    '$meta': 'vectorSearchScore'
                }

            }
        }
    ]
    result = RAGSegment.objects().aggregate(pipeline)
    print("context_docs",list(result))
    return list(result)