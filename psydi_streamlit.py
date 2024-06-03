import streamlit as st
import os
import os.path
from dotenv import load_dotenv

from llama_index.llms.azure_openai import AzureOpenAI
from zhihu_search_tool import ZhihuSearchToolSpec
from llama_index.tools.google import GoogleSearchToolSpec
from llama_index.core.tools.tool_spec.load_and_search.base import LoadAndSearchToolSpec
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core import Settings
from llama_index.core.agent import ReActAgent
from llama_index.core.response.pprint_utils import pprint_response
from llama_index.core.base.llms.types import ChatMessage

load_dotenv()
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ["LLAMA_INDEX_CACHE_DIR"] = '/mnt/nfs/renjiyuan/HF_cache'
os.environ['HF_HOME'] = '/mnt/nfs/renjiyuan/HF_cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/mnt/nfs/renjiyuan/HF_cache'

sys_prompt = """
参考飞书文档
"""

user_prompt_pre = """
下面是我给出的回答，如果回答中或者回答对应的选项中有任何您不知道的词汇/特定概念，请逐个查询互联网之后，结合查询到的内容进行后续问答：
"""

# Setup AzureOpenAI Agent
llm = AzureOpenAI(
    engine="gpt-4",
    model="gpt-4",
    temperature=0.0
)
# or choose seepseek
# from llama_index.llms.openai import OpenAI
# llm = OpenAI(
#     model="deepseek-chat",
#     temperature=0.0,
#     api_base="",
#     api_key=""
# )

# test llm api
# completion_response = llm.complete("To infinity, and")
# print('Test api:\n', completion_response)

# setup SearchToolSpec
# key and engine could be found in https://developers.google.com/custom-search/v1/introduction?hl=zh-cn
engine = ''
key = ''
zhihu_spec = ZhihuSearchToolSpec(key=key, engine=engine)
google_spec = GoogleSearchToolSpec(key=key, engine=engine)

# prepare tools
tools = []
tools += LoadAndSearchToolSpec.from_defaults(
    zhihu_spec.to_tool_list()[0],
).to_tool_list()
tools += LoadAndSearchToolSpec.from_defaults(
    google_spec.to_tool_list()[0],
).to_tool_list()

# setting the embed model, for default embed model is openai.
# If you have the key of openai, you can set it in the env and skip this step.
Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")
Settings.llm = llm

agent = ReActAgent.from_tools(tools, llm=llm, verbose=True)

st.title("PsyDI-Google")
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        ChatMessage(role="system", content=sys_prompt)
    ]
    response = agent.chat(sys_prompt)
    st.session_state.messages.append(ChatMessage(role="assistant", content=response.response))
    
if prompt := st.chat_input("Your question"): 
    prompt = f"""下面是我给出的回答，如果回答中或者回答对应的选项中有任何您不知道的词汇/特定概念，请逐个查询互联网之后，结合查询到的内容进行后续问答：“{prompt}”"""
    st.session_state.messages.append(ChatMessage(role="user", content=prompt))

for message in st.session_state.messages: 
    if message.dict()["role"] != "system":
        with st.chat_message(message.dict()["role"]):
            st.write(message.dict()["content"])
        
if st.session_state.messages[-1].dict()["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = agent.chat(message=prompt, chat_history=st.session_state.messages)
            st.write(response.response)
            pprint_response(response, show_source=True)
            message = ChatMessage(role="assistant", content=response.response)
            st.session_state.messages.append(message)