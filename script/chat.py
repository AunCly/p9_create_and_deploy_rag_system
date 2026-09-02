from dotenv import load_dotenv
from script.rag import EventsRag as Rag

load_dotenv()

rag = Rag()

prompt = "Quels sont les événements à propos de l'architecture ?"

answer = rag.answer(prompt)

print(answer.content)