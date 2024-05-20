"""Use Google Search to search for content in Zhihu."""

import urllib.parse
from typing import Optional

import requests
from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec

QUERY_URL_TMPL = (
    "https://www.googleapis.com/customsearch/v1?key={key}&cx={engine}&q={query}"
)


class ZhihuSearchToolSpec(BaseToolSpec):
    """
    Zhihu Search tool spec. \
    Zhihu is a Chinese Q&A website where users can ask questions, offer answers, and engage in discussions on a wide range of topics.
    """

    spec_functions = ["zhihu_search"]

    def __init__(self, key: str, engine: str, num: Optional[int] = None) -> None:
        """Initialize with parameters."""
        self.key = key
        self.engine = engine
        self.num = num

    def zhihu_search(self, query: str):
        """
        Make a query to the Google search engine to receive a list of results.

        Args:
            query (str): The query to be passed to Google search.
            num (int, optional): The number of search results to return. Defaults to None.

        Raises:
            ValueError: If the 'num' is not an integer between 1 and 10.
        """
        site = "zhihu.com"
        query_with_site = f"site:{site} {query}"
        url = QUERY_URL_TMPL.format(
            key=self.key, engine=self.engine, query=urllib.parse.quote_plus(query_with_site)
        )

        if self.num is not None:
            if not 1 <= self.num <= 10:
                raise ValueError("num should be an integer between 1 and 10, inclusive")
            url += f"&num={self.num}"

        response = requests.get(url)
        return [Document(text=response.text)]