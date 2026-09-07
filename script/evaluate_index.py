import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
import script.utils as utils
import warnings

from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_community")

base_path = Path(os.path.dirname(__file__)).parent
vectorstore = FAISS.load_local(
    folder_path=str(base_path / "data" / "raw" / "faiss_langchain_index"),
    embeddings=GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        max_retries=5,
    ),
    allow_dangerous_deserialization=True
)

# 2. Check basic properties of the index
print(f"Total vectors in index: {vectorstore.index.ntotal}")
print(f"Vector dimension: {vectorstore.index.d}")
print(f"Is trained: {vectorstore.index.is_trained}")

query = "Evénement a propos de l'architecture"

query_embed = utils.embed_text(query)

docs_and_scores = vectorstore.similarity_search_with_score(query, k=5)

print("Nearest neighbors indices:", [i for i, _ in docs_and_scores])
print("Corresponding distances:", [score for _, score in docs_and_scores])

for idx, (doc, score) in enumerate(docs_and_scores):
    print(f"Document {idx}: {doc.page_content}")
    print('------------------')

