import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_community")

import script.embedding as embedding

base_path = Path(os.path.dirname(__file__)).parent
vectorstore = FAISS.load_local(
    folder_path=str(base_path / "data" / "raw" / "faiss_langchain_index"),
    embeddings=MistralAIEmbeddings(
        model="mistral-embed",
        max_retries=5,
    ),
    allow_dangerous_deserialization=True
)

# 2. Check basic properties of the index
print(f"Total vectors in index: {vectorstore.index.ntotal}")
print(f"Vector dimension: {vectorstore.index.d}")
print(f"Is trained: {vectorstore.index.is_trained}")

query = "Evénement a propos de l'architecture"

query_embed = embedding.embed_text(query)

docs_and_scores = vectorstore.similarity_search_with_score(query, k=5)

print("Nearest neighbors indices:", [i for i, _ in docs_and_scores])
print("Corresponding distances:", [score for _, score in docs_and_scores])

for idx, (doc, score) in enumerate(docs_and_scores):
    print(f"Document {idx}: {doc.page_content}")
    print('------------------')