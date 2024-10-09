from src.models.rag import RAGSegment
from src.utils.embeddings import generate_embeddings

def mongodb_vector_search(vector_index, text, context, k):

    query_vector = generate_embeddings(text)

    #Aqui está dando problemas de timeout devido a ausencia da collection
    num_candidates = RAGSegment.objects.count()

    print(f"num_candidates: {num_candidates}")
    pipeline = [
        {
            '$vectorSearch': {
                'exact':True,
                'index':vector_index,
                'path': 'embedding',
                'filter': {
                    'context':context
                },
                'queryVector': query_vector,
                #'numCandidates': num_candidates,
                'limit': k if k <= num_candidates else num_candidates
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
    print("result", result)
    return list(result)