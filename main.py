from fastapi import FastAPI
from pydantic import BaseModel

from llm_utils import get_db, load_documents, get_text_splitter
from llm_utils import get_llm, get_query_rephraser, get_retriever, get_qa_langchain

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders.csv_loader import CSVLoader



text_splitter = get_text_splitter(chunk_size=400, chunk_overlap=0)
documents = load_documents('docs/', text_splitter, loaders={'.pdf': PyPDFLoader, '.csv': CSVLoader})
# vectorstore = get_db(documents, embedder_name='ai-forever/sbert_large_nlu_ru')
vectorstore = get_db(documents, embedder_name='ai-forever/ruElectra-large')
# vectorstore = get_db(documents)

llm = get_llm(n_ctx=5_000)
# rephraser = get_query_rephraser(llm)
retriever = get_retriever(vectorstore, {"k": 4})
chain = get_qa_langchain(llm, retriever)

app = FastAPI()

class Message(BaseModel):
    message: str
    user_id: str


@app.post("/message")
async def handle_message(data: Message):
    return {"response": f"{chain.invoke({'question': data.message})}"}