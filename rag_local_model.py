import os
from typing import List, Optional

from llama_index.llms.huggingface import (
    HuggingFaceInferenceAPI,
    HuggingFaceLLM,
)
from llama_index.core import ServiceContext, set_global_service_context
from llama_index.core.embeddings import resolve_embed_model
import os.path
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings
)


os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# model_name = '/mnt/nfs/renjiyuan/HF_cache/hub/models--meta-llama--Llama-2-13b-hf/snapshots/5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1'
model_name = 'meta-llama/Llama-2-7b-chat-hf'
model_kwargs = {'cache_dir': '/mnt/nfs/renjiyuan/HF_cache'}
llm = HuggingFaceLLM(
    model_name=model_name,
    tokenizer_name=model_name,
    context_window=4096,
    model_kwargs=model_kwargs
    )

completion_response = llm.complete("To infinity, and")
print('Test api:\n', completion_response)

# service_context = ServiceContext.from_defaults(
#     chunk_size=1024,
#     llm=llm,
#     embed_model=f'local:{model_name}'
# )

Settings.embed_model = resolve_embed_model(f'local:{model_name}')

# ollama
Settings.llm = llm

# check if storage already exists
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

# Either way we can now query the index
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
print(response)