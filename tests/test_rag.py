import os

from langchain_core.messages import AIMessage

from script.create_faiss_index import create_faiss_index
from script.rag import EventsRag as Rag

class TestRag:

    def test_rag_initialization(self):
        rag_model = Rag()
        assert rag_model is not None
        assert isinstance(rag_model, Rag)

    def test_rag_answer(self):
        rag_model = Rag()
        prompt = "A quelle date peut-on voir Dany-Io le sculpteur de pierre ?"
        answer, documents = rag_model.answer(prompt)

        assert answer is not None
        assert isinstance(answer, AIMessage)
        assert isinstance(documents, list)

    def test_build_index(self):

        faiss_path = "data/raw/faiss_langchain_index/index.faiss"
        if os.path.exists(faiss_path):
            os.remove(faiss_path)

        pkl_path = "data/raw/faiss_langchain_index/index.pkl"
        if os.path.exists(pkl_path):
            os.remove(pkl_path)

        create_faiss_index()

        index_path = "data/raw/faiss_langchain_index/index.faiss"
        assert os.path.exists(index_path)

        faiss_path = "data/raw/faiss_langchain_index/index.faiss"
        assert os.path.isfile(faiss_path)