from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.document_loaders.csv_loader import CSVLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.prompts import ChatPromptTemplate, PromptTemplate
from operator import itemgetter
from langchain.schema.output_parser import StrOutputParser

from langchain.schema.prompt_template import format_document




"""
Documents and database
"""

# def process_docs(docs):
#     prompt = PromptTemplate.from_template("{page_content}\n")
#     return [format_document(doc, prompt) for doc in docs]

def get_db(chunks, embedder_name='cointegrated/LaBSE-en-ru'):
    embeddings_model = HuggingFaceEmbeddings(model_name=embedder_name)
    db = Chroma.from_documents(chunks, embeddings_model)
    return db

def load_documents(
        docs_path, text_splitter=None, 
        loaders={
        '.pdf': PyPDFLoader,
        '.csv': CSVLoader,},
        loader_kwargs=None
    ):
    def create_directory_loader(file_type, directory_path):
        return DirectoryLoader(
            path=directory_path,
            glob=f"**/*{file_type}",
            loader_cls=loaders[file_type],
            loader_kwargs=loader_kwargs
        )
    pdf_loader = create_directory_loader('.pdf', docs_path)
    csv_loader = create_directory_loader('.csv', docs_path)
    if text_splitter:
        pdf_documents = pdf_loader.load_and_split(text_splitter=text_splitter)
        csv_documents = csv_loader.load_and_split(text_splitter=text_splitter)
    else:
        pdf_documents = pdf_loader.load()
        csv_documents = csv_loader.load()
    return pdf_documents + csv_documents

def get_text_splitter(chunk_size=1000, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(        
        chunk_size = chunk_size,
        chunk_overlap  = chunk_overlap,
        length_function = len
    )
    return text_splitter


"""
LLM, prompt and langchain
"""

def get_llm(model_path='models/llama-2-7b-chat.Q4_K_M.gguf', n_ctx=4096):
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = LlamaCpp(
        model_path=model_path,
        temperature=0.01, #0.75,
        max_tokens=min(n_ctx, 4000),
        n_ctx=n_ctx,
        top_p=1,
        callback_manager=callback_manager,
        verbose=True, # Verbose is required to pass to the callback manager
    )
    return llm

def get_lang_chain(model, retriever):
    template = \
"""<<SYS>>Твоя роль: Виртуальный ассистент банка Тинькофф, отвечающий на фактологические вопросы. Язык для ответа: только русский.  
Игнорируй вопросы, на которые не можешь ответить корректно. Отвечай на основе найденных документов, но не упоминай сами документы.<<\SYS>>
[INST]Вопрос: {question}[/INST]
<<SYS>>Найденные документы: {context}<<\SYS>>
Ответ:
"""
    prompt = ChatPromptTemplate.from_template(template)
    chain = {
        "context": itemgetter("question") | retriever, 
        "question": itemgetter("question") 
    } | prompt | model | StrOutputParser()

    return chain

