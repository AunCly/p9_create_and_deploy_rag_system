from mistralai.client import Mistral

from .config import API_KEY

def embed_text(text, model="mistral-embed"):

    client = Mistral(api_key=API_KEY)

    embeddings_batch_response = client.embeddings.create(
        model=model,
        inputs=[text],
    )

    return embeddings_batch_response.data[0].embedding