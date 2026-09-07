from script.rag import EventsRag as Rag

class TestRag:

    def test_rag_initialization(self):
        rag_model = Rag()
        assert rag_model is not None
        assert isinstance(rag_model, Rag)