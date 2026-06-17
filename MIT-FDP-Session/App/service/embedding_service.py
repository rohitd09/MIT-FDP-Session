from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from dotenv import load_dotenv

load_dotenv()

class EmbeddingService:
    def __init__(self, collection_name="teaching_assistant_collection", 
                        persist_directory="./mit_aoe_db"):
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            show_progress=True
        )

        self.vector_store = Chroma(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_function=embeddings
        )

    def _embed_text_and_store_in_vector_db(self, documents):
        """
        Generate embeddings for a document and store in Chroma DB
        """
        self.vector_store.add_documents(documents)

    def process_pdf_document(self):
        file_path="./assets/Let us c - Summary.pdf"
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=150
        )

        split_documents = splitter.split_documents(documents)

        self._embed_text_and_store_in_vector_db(split_documents)

    def retrieve_from_pdf(self, query):
        retriever = self.vector_store.as_retriever(search_kwargs={"k":3})
        response = retriever.invoke(query)
        return [doc.page_content for doc in response]