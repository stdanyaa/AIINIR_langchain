from fastapi import FastAPI
from pydantic import BaseModel

from llm_utils import get_llm, get_db, load_documents, get_text_splitter, get_lang_chain, is_message_safe

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders.csv_loader import CSVLoader



llm = get_llm(n_ctx=5_000)
text_splitter = get_text_splitter(chunk_size=1000, chunk_overlap=200)
documents = load_documents('docs/', text_splitter, loaders={'.pdf': PyPDFLoader, '.csv': CSVLoader})
# retriever = get_db(documents, embedder_name='ai-forever/ruElectra-large').as_retriever(k=2)
retriever = get_db(documents).as_retriever(k=2)
chain = get_lang_chain(llm, retriever)

app = FastAPI()

class Message(BaseModel):
    message: str
    user_id: str


@app.post("/message")
async def handle_message(data: Message):
    if is_message_safe(data.message):
        return {"response": f"{chain.invoke({'question': data.message})}"}
    else:
        return {"response": "Пожалуйста, напишите корректный запрос"}