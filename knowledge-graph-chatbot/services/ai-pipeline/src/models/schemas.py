"""Shared schema models for chat, ingestion, extraction, and graph retrieval."""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class TokenType(str, Enum):
    TEXT = "TEXT"
    CITATION = "CITATION"
    GRAPH_UPDATE = "GRAPH_UPDATE"
    ERROR = "ERROR"
    DONE = "DONE"


class IngestStage(str, Enum):
    CHUNKING = "CHUNKING"
    ENTITY_EXTRACTION = "ENTITY_EXTRACTION"
    RELATION_EXTRACTION = "RELATION_EXTRACTION"
    EMBEDDING_GENERATION = "EMBEDDING_GENERATION"
    GRAPH_STORAGE = "GRAPH_STORAGE"
    COMPLETED = "COMPLETED"


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatOptions(BaseModel):
    temperature: float = 0.7
    max_tokens: int = 1024
    include_graph_citation: bool = True
    stream_subgraph: bool = True


class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = Field(default_factory=list)
    session_id: str = ""
    options: ChatOptions = Field(default_factory=ChatOptions)


class ChatToken(BaseModel):
    text: str
    type: TokenType


class GraphNode(BaseModel):
    id: str
    label: str
    name: str
    properties: Dict[str, str] = Field(default_factory=dict)
    confidence: float = 0.0


class GraphEdge(BaseModel):
    id: str
    source_id: str
    target_id: str
    relation_type: str
    weight: float = 0.0
    source_document: str = ""


class SubgraphResult(BaseModel):
    nodes: List[GraphNode] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)
    total_nodes_visited: int = 0
    traversal_time_ms: float = 0.0


class VectorSearchResult(BaseModel):
    node_id: str
    distance: float
    node: Optional[GraphNode] = None


class Document(BaseModel):
    id: str
    content: str
    title: str = ""
    metadata: Dict[str, str] = Field(default_factory=dict)


class TextChunk(BaseModel):
    id: str
    text: str
    document_id: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: Dict[str, str] = Field(default_factory=dict)


class ExtractedEntity(BaseModel):
    id: str = ""
    text: str
    label: str
    start_char: int
    end_char: int
    confidence: float = 0.0


class ExtractedRelation(BaseModel):
    head_text: str
    head_label: str
    relation: str
    tail_text: str
    tail_label: str
    confidence: float = 0.0
    source_sentence: str = ""


class EmbeddingResult(BaseModel):
    vector: List[float]
    dimensions: int
    source_text: str = ""


class IngestProgress(BaseModel):
    stage: IngestStage
    progress: float
    message: str
    entities_found: int = 0
    relations_found: int = 0
    chunks_processed: int = 0
    total_chunks: int = 0

