import os
from typing import List, Optional

from llama_index.llms.huggingface import (
    HuggingFaceInferenceAPI,
    HuggingFaceLLM,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
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
os.environ["LLAMA_INDEX_CACHE_DIR"] = '/mnt/nfs/renjiyuan/HF_cache'
os.environ['HF_HOME'] = '/mnt/nfs/renjiyuan/HF_cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/mnt/nfs/renjiyuan/HF_cache'

# model_name = '/mnt/nfs/renjiyuan/HF_cache/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590'
model_name = '/mnt/nfs/lixueyan/LLM/llama-2-7b-hf'
# model_name = 'meta-llama/Llama-2-7b-chat-hf'
# model_name = '/mnt/nfs/renjiyuan/HF_cache/hub/models--meta-llama--Llama-2-13b-hf/snapshots/5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1'
# model_kwargs = {'cache_dir': '/mnt/nfs/renjiyuan/HF_cache'}
llm = HuggingFaceLLM(
    model_name=model_name,
    tokenizer_name=model_name,
    # max_new_tokens=2048,
    context_window=2048,
    # model_kwargs=model_kwargs
    )

completion_response = llm.complete("To infinity, and")
print('Test api:\n', completion_response)

# method 1: set llm through global service context
# service_context = ServiceContext.from_defaults(
#     chunk_size=1024,
#     llm=llm,
#     embed_model=f'local:{model_name}'
# )

# method 2: set llm through Settings
# Settings.embed_model = HuggingFaceEmbedding(model_name=model_name)
Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")
Settings.llm = llm

documents = SimpleDirectoryReader("data").load_data()
# index = VectorStoreIndex.from_documents(documents, service_context=service_context)
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
print('begin query:\n')
# response = query_engine.query("What did the author do growing up?")
response = query_engine.query("What is the author's name?")
print('response:\n',response)