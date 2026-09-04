from script.rag import EventsRag as Rag

class TestRag:

    def __init__(self):
        self.questions = [
            'Quels sont les événements à propos de l\'architecture ?',
            'Quels est le prochain concert à venir à la Sirène ?',
            'A quelle heure commence le spectacle Premier regard sur L’homme qui danse ?',
            'Quels sont les évènement gratuit en Septembre 2027 ?'
            'Quand se passe le festival de la BD d\'Angoulême en 2026 ?',
        ]

    def test_rag_initialization(self):
        rag_model = Rag()
        assert rag_model is not None
        assert isinstance(rag_model, Rag)