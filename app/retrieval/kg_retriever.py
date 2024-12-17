"""
This module provides implementations for a Knowledge Graph retriever and a corresponding query engine.
The retriever uses a knowledge graph to extract relevant information chunks based on the highest degree nodes,
ignoring the input query.

Classes:
    KnowledgeGraphRetriever: Retrieves relevant text snippets based on a query
    using knowledge graph relationships.
    KGRetrievalQueryEngine: Initializes and configures a knowledge graph retriever query engine.
"""

import time
from langchain_core.language_models import BaseLLM
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import NodeWithScore, TextNode


class KnowledgeGraphRetriever(BaseRetriever):
    """
    An extractive retriever that uses a knowledge graph to retrieve relevant text snippets based on the
    nodes with the highest degree in the graph. The retriever ignores the input query and instead uses
    the structure of the knowledge graph to find relevant information.

    Attributes:
        index: The knowledge graph index used for retrieval.
        embedding_model: The embedding model used for document representation.
        top_k (int): The number of top nodes to retrieve based on their degree in the graph.
        verbose (bool): Whether to print verbose output.

    Methods:
        _retrieve(query_bundle) -> list:
            Processes a query and returns a list of extracted sentences as NodeWithScore.
    """

    def __init__(
            self,
            index,
            embed_model,
            verbose,
            top_k=3
    ):
        """
        Initializes the KnowledgeGraphRetriever with the specified parameters.

        Parameters:
            index: The knowledge graph index used for retrieval.
            embed_model: The embedding model used for document representation.
            verbose (bool): Whether to print verbose output.
            top_k (int, optional): The number of top nodes to retrieve based on their degree in the graph.
                                   Defaults to 3.
        """
        super().__init__()
        self.index = index
        self.embedding_model = embed_model
        self.top_k = top_k
        self.verbose = verbose

    def _retrieve(self, query_bundle):
        """
        Retrieves relevant chunks by ignoring the query_bundle and using information from a Knowledge Graph instead.
        We use the top_k nodes with highest degree in the graph as the most relevant chunks. A chunk is constructed
        by extracting the edge relationships of a node.

        Parameters:
            query_bundle (object): A query bundle that is necessary to match the signature of the parent class.

        Returns:
            list: A list of NodeWithScore objects representing the most relevant chunks along with their scores.
        """
        if self.verbose:
            start_time = time.time()

        # retrieve graph from index
        g = self.index.get_networkx_graph()

        if self.verbose:
            end_time = time.time()
            duration = end_time - start_time
            print(f"[+] Done retrieving Knowledge Graph. Time: {duration:.6f}s")

        if self.verbose:
            start_time = time.time()

        # get nodes with the highest degree
        node_degrees = dict(g.degree())
        top_k_nodes = sorted(node_degrees, key=node_degrees.get, reverse=True)[:self.top_k]

        if self.verbose:
            end_time = time.time()
            duration = end_time - start_time
            print(f"[+] Done obtaining highest-degree nodes from graph. Time: {duration:.6f}s")

        if self.verbose:
            start_time = time.time()

        # extract edge relationships
        top_k_nodes_edges_info = {}
        for node in top_k_nodes:
            edges = g.edges(node, data=True)
            edges_info = []
            for edge in edges:
                target_node = edge[1] if edge[0] == node else edge[0]
                edge_label = edge[2].get('label', 'has_relation')
                edges_info.append({
                    "source_node": node,
                    "edge_label": edge_label,
                    "target_node": target_node
                })
            top_k_nodes_edges_info[node] = edges_info

        if self.verbose:
            end_time = time.time()
            duration = end_time - start_time
            print(f"[+] Done obtaining edge relationships. Time: {duration:.6f}s")

        if self.verbose:
            start_time = time.time()

        summarized_strings = []
        for node, edges_info in top_k_nodes_edges_info.items():
            edge_descriptions = [
                f"{edge_info['source_node']} {edge_info['edge_label']} {edge_info['target_node']}"
                for edge_info in edges_info
            ]
            summarized_string = ". ".join(edge_descriptions) + "."
            summarized_strings.append(NodeWithScore(node=TextNode(text=summarized_string)))

        if self.verbose:
            end_time = time.time()
            duration = end_time - start_time
            print(f"[+] Done creating chunks from edges. Time: {duration:.6f}s")

        return summarized_strings


class KGRetrievalQueryEngine:
    """
    Class representing a query engine for Knowledge Graph Retrieval.

    Attributes:
        query_engine (RetrieverQueryEngine): The configured query engine for knowledge graph retrieval.

    Methods:
        __init__(llm: BaseLLM, index, embed_model, verbose=False):
            Initializes the KGRetrievalQueryEngine with the specified parameters.
    """

    def __init__(
            self,
            llm: BaseLLM,
            index,
            embed_model,
            verbose=False
    ):
        """
        Initializes the KGRetrievalQueryEngine with the specified parameters.

        Parameters:
            llm (BaseLLM): The language model used for query generation.
            index: The index used for topic retrieval. For KGRetrieval, only KnowledgeGraphIndex is supported.
            embed_model: The embedding model used for document representation.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
        """
        retriever = KnowledgeGraphRetriever(index, embed_model=embed_model, verbose=verbose)
        self.query_engine = RetrieverQueryEngine.from_args(retriever=retriever, llm=llm)
