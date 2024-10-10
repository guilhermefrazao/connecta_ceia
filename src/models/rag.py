from mongoengine import (
    Document,
    StringField,
    FloatField,
    ListField,
    ObjectIdField,
    DictField
)

from bson import ObjectId
import os

class RAGSegment(Document):
    id = ObjectIdField(primary_key=True, default=lambda: ObjectId())
    context = StringField()
    text = ListField(DictField())
    text_embedding = StringField()
    source_type = StringField()
    embedding = ListField(FloatField(), required=False)
    instructions = StringField()


    #trocar nome da collection
    meta = {
        "collection": "connecta_rag",
    }

    def to_dict(self):
        return {
            "id":self.str(self.id),
            "context":self.context,
            "text":self.text,
            "source":self.source_type
        }   
    
    @classmethod
    def create_vector_index(cls):
        collection = RAGSegment._get_collection()
        indexes = collection.index_information()

        index_name = os.getenv('CONNECTA_VECTOR_INDEX_NAME', 'vector_index')

        if index_name not in indexes:
            from pymongo.operations import SearchIndexModel
            search_index_model = SearchIndexModel(
                type='vectorSearch',
                definition={
                    "mappings": {
                        "dynamic": False
                    },
                    "fields": [
                        {
                            "type": "vector",
                            "path": "embedding",
                            "numDimensions": 1536,
                            "similarity": "euclidean"
                        },
                        {
                            "type": "filter",
                            "path": "context"
                        },
                    ]
                },
                name=index_name,
            )
            result = collection.create_search_index(model=search_index_model)
            print(result)
        else:
            print('vector index already exists')
    
    @classmethod
    def list_search_indexes(cls):
        collection = RAGSegment._get_collection()
        search_index = list(collection.list_search_indexes(os.getenv('CONNECTA_VECTOR_INDEX_NAME', 'vector_index')))
        print(f"Indices de Busca MongoDB:{search_index}")