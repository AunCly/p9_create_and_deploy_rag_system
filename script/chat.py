from dotenv import load_dotenv
from script.rag import EventsRag as Rag

load_dotenv()

rag = Rag()

prompt = "A quelle date peut-on voir Dany-Io le sculpteur de pierre ?"

answer, documents = rag.answer(prompt)

print(answer.content[0]['text'])