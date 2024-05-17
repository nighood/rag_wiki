from dotenv import load_dotenv
import os
load_dotenv()
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ["LLAMA_INDEX_CACHE_DIR"] = '/mnt/nfs/renjiyuan/HF_cache'
os.environ['HF_HOME'] = '/mnt/nfs/renjiyuan/HF_cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/mnt/nfs/renjiyuan/HF_cache'

# Setup AzureOpenAI Agent
from llama_index.llms.azure_openai import AzureOpenAI

llm = AzureOpenAI(
    engine="gpt-4",
    model="gpt-4",
    temperature=0.0
)

# test llm api
completion_response = llm.complete("To infinity, and")
print('Test api:\n', completion_response)

# from llama_index.core.tools.download import download_tool
# from llama_index.core.tools.tool_spec.load_and_search.base import LoadAndSearchToolSpec
from baidu_search_tool import BaiduSearchToolSpec, baidu_search

# # setup GoogleSearchToolSpec
# baidu_spec = BaiduSearchToolSpec()

# # prepare tools
# tools = LoadAndSearchToolSpec.from_defaults(
#     baidu_spec.to_tool_list()[0],
# ).to_tool_list()

from llama_index.core.tools import FunctionTool
tools = FunctionTool.from_defaults(fn=baidu_search)


from llama_index.core.embeddings import resolve_embed_model
from llama_index.core import Settings

# setting the embed model, for default embed model is openai.
# If you have the key of openai, you can set it in the env and skip this step.
Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")
Settings.llm = llm

from llama_index.core.agent import AgentRunner,ReActAgent
# this method will automatically pick the best agent depending on the LLM
agent = AgentRunner.from_llm([tools], llm=llm, verbose=True)

# or
# from llama_index.core.agent import ReActAgent
# agent = ReActAgent.from_tools(tools, llm=llm, verbose=True)

# test 
agent.chat("什么是MBTI？")