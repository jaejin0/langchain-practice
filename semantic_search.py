import os
from dotenv import load_dotenv

from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.runnables import chain

''' initialize '''

load_dotenv()


''' retrieve '''

# format
documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
]


# loader
file_path = "./resources/Multi_Agent_Checkins_DARS.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()
print(docs[0].metadata)


# splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)
print(len(all_splits))


# embeddings
embeddings = OllamaEmbeddings(model="llama3.2")

vector_1 = embeddings.embed_query(all_splits[0].page_content)
vector_2 = embeddings.embed_query(all_splits[1].page_content)
assert len(vector_1) == len(vector_2)
print(f"Generated vectors of length {len(vector_1)}\n")
print(vector_1[:10])


# vector store
vector_store = InMemoryVectorStore(embeddings)
ids = vector_store.add_documents(documents=all_splits)
results = vector_store.similarity_search(
    "Gather information about the cost function"
)
# print(results[0])


# retriever
@chain
def retriever(query: str) -> List[Document]:
    return vector_store.similarity_search(query, k=1)

print(retriever.batch(
    [
        "Gather information about the cost function",
        "When was this paper published?",
    ],
))




