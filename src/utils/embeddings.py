from openai import OpenAI

def generate_embeddings(text: str):
    client = OpenAI()
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model='text-embedding-3-small').data[0].embedding