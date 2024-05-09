import streamlit as st
import os
import os.path

from dotenv import load_dotenv
from llama_index.core import Settings, download_loader, VectorStoreIndex, load_index_from_storage, ServiceContext
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.legacy.embeddings import AzureOpenAIEmbedding
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.response.pprint_utils import pprint_response
from llama_index.llms.openai import OpenAI
from llama_index.readers.wikipedia import WikipediaReader

load_dotenv()
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ["LLAMA_INDEX_CACHE_DIR"] = '/mnt/nfs/renjiyuan/HF_cache'
os.environ['HF_HOME'] = '/mnt/nfs/renjiyuan/HF_cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/mnt/nfs/renjiyuan/HF_cache'

storage_path = "./vectorstore"

# llm = OpenAI(temperature=0.1, model="gpt-4-turbo-preview")
llm = AzureOpenAI(
    engine="gpt-4",
    model="gpt-4",
    temperature=0.0
)
# service_context = ServiceContext.from_defaults(llm=llm, embed_model="local:BAAI/bge-small-en-v1.5")

# Settings.embed_model = AzureOpenAIEmbedding(
#     model="text-embedding-ada-002",
#     deployment_name="text-embedding-ada-002",
#     )
Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")
Settings.llm = llm

WikipediaReader = download_loader("WikipediaReader")

loader = WikipediaReader()
documents = loader.load_data(pages=['Star Wars Movie', 'Star Trek Movie'])
index = VectorStoreIndex.from_documents(documents)
# index.storage_context.persist(persist_dir=storage_path)


# st.title("Ask the Wiki On Star Wars")
# if "messages" not in st.session_state.keys(): 
#     st.session_state.messages = [
#         {"role": "assistant", "content": "Ask me a question !"}
#     ]

chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)
prompt = "give the list all the star wars episode"
# retrive_response = chat_engine.retrieve("give the list all the star wars episode")
prompt = "give a table of all the star wars episode, with year and the main cast"
response = chat_engine.chat(prompt)
print(response.response)

# if prompt := st.chat_input("Your question"): 
#     st.session_state.messages.append({"role": "user", "content": prompt})

# for message in st.session_state.messages: 
#     with st.chat_message(message["role"]):
#         st.write(message["content"])
        
# if st.session_state.messages[-1]["role"] != "assistant":
#     with st.chat_message("assistant"):
#         with st.spinner("Thinking..."):
#             response = chat_engine.chat(prompt)
#             st.write(response.response)
#             pprint_response(response, show_source=True)
#             message = {"role": "assistant", "content": response.response}
#             st.session_state.messages.append(message) 