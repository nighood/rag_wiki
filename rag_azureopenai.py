from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.core.embeddings import resolve_embed_model
import os
from llama_index.legacy.embeddings import AzureOpenAIEmbedding

os.environ["OPENAI_API_KEY"] = "your api key"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://api.openai.com"
os.environ["OPENAI_API_VERSION"] = "your version"

llm = AzureOpenAI(
    engine="gpt-4",
    model="gpt-4",
    temperature=0.0
)
completion_response = llm.complete("To infinity, and")
print('Test api:\n', completion_response)

Settings.embed_model = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name="text-embedding-ada-002",
    )
Settings.llm = llm

PERSIST_DIR = "./storage"
if not os.path.exists(PERSIST_DIR):
    # load the documents and create the index
    documents = SimpleDirectoryReader("data").load_data()
    # index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    index = VectorStoreIndex.from_documents(documents)
    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

# ---------query index----------
# Either way we can now query the index
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
print(response)