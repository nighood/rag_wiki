import streamlit as st
import os
import os.path
from dotenv import load_dotenv

from llama_index.llms.azure_openai import AzureOpenAI
from zhihu_search_tool import ZhihuSearchToolSpec
from llama_index.core.tools.tool_spec.load_and_search.base import LoadAndSearchToolSpec
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core import Settings
from llama_index.core.agent import ReActAgent
from llama_index.core.response.pprint_utils import pprint_response

load_dotenv()
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ["LLAMA_INDEX_CACHE_DIR"] = '/mnt/nfs/renjiyuan/HF_cache'
os.environ['HF_HOME'] = '/mnt/nfs/renjiyuan/HF_cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/mnt/nfs/renjiyuan/HF_cache'

# Setup AzureOpenAI Agent
llm = AzureOpenAI(
    engine="gpt-4",
    model="gpt-4",
    temperature=0.0
)
# test llm api
completion_response = llm.complete("To infinity, and")
# print('Test api:\n', completion_response)

# setup SearchToolSpec
# key and engine could be found in https://developers.google.com/custom-search/v1/introduction?hl=zh-cn
engine = '4690d06089adf4627'
key = 'AIzaSyBSu-P_vk0i3Be7E5TMB_q6L-i5vR61u_s'
zhihu_spec = ZhihuSearchToolSpec(key=key, engine=engine)

# prepare tools
tools = LoadAndSearchToolSpec.from_defaults(
    zhihu_spec.to_tool_list()[0],
).to_tool_list()

# setting the embed model, for default embed model is openai.
# If you have the key of openai, you can set it in the env and skip this step.
Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")
Settings.llm = llm

agent = ReActAgent.from_tools(tools, llm=llm, verbose=True)
# agent.chat("DILab 是谁?")

st.title("Ask me a question!")
if "messages" not in st.session_state.keys(): 
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question !"}
    ]

if prompt := st.chat_input("Your question"): 
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: 
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = agent.chat(prompt)
            st.write(response.response)
            pprint_response(response, show_source=True)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) 