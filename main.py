from fastapi import FastAPI
from pydantic import BaseModel

from llm_utils import get_llm, get_db, load_documents, get_text_splitter, get_lang_chain

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders.csv_loader import CSVLoader



text_splitter = get_text_splitter(chunk_size=1200, chunk_overlap=200)
documents = load_documents('docs/', text_splitter, loaders={'.pdf': PyPDFLoader, '.csv': CSVLoader})
# retriever = get_db(documents, embedder_name='ai-forever/sbert_large_nlu_ru').as_retriever(search_kwargs={"k": 3})
retriever = get_db(documents).as_retriever(search_kwargs={"k": 2})

llm = get_llm(n_ctx=5_000)
chain = get_lang_chain(llm, retriever)

app = FastAPI()

class Message(BaseModel):
    message: str
    user_id: str


@app.post("/message")
async def handle_message(data: Message):
    return {"response": f"{chain.invoke({'question': data.message})}"}