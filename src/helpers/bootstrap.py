"""Build every runtime component from Settings, in one place.

Used by the FastAPI lifespan, the stdio MCP server, Telegram polling mode
and the eval scripts, so they all run exactly the same agent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from dotenv import load_dotenv

from agent import Agent
from helpers.config import Settings, get_settings
from stores.llm.LLMProviderFactory import LLMProviderFactory
from stores.llm.pricing import load_price_overrides
from stores.mcp import McpRegistry
from stores.vectordb.VectorDBProviderFactory import VectorDBProviderFactory
from tools import DoctorTools, web_search
from tools.skills import SkillLibrary

logger = logging.getLogger(__name__)


@dataclass
class Services:
    config: Settings
    llm: Any = None
    embedding: Any = None
    vector_db: Any = None
    doctor_tools: Optional[DoctorTools] = None
    skills: Optional[SkillLibrary] = None
    mcp_registry: Optional[McpRegistry] = None
    agent: Optional[Agent] = None
    errors: dict = field(default_factory=dict)

    def close(self) -> None:
        if self.mcp_registry is not None:
            self.mcp_registry.close()
        if self.vector_db is not None:
            self.vector_db.disconnect()


def build_services(config: Optional[Settings] = None, connect_mcp: bool = True) -> Services:
    load_dotenv()  # third-party SDKs (Tavily) read keys from os.environ
    config = config or get_settings()
    services = Services(config=config)
    load_price_overrides(config.MODEL_PRICES_JSON)

    llm_factory = LLMProviderFactory(config=config)
    services.llm = llm_factory.create(config.GENERATION_BACKEND)
    services.embedding = llm_factory.create(config.EMBEDDING_BACKEND)
    if services.llm is None:
        services.errors["llm"] = f"Unknown GENERATION_BACKEND {config.GENERATION_BACKEND!r}"
    else:
        services.llm.set_generation_model(config.GENERATION_MODEL_ID)
    if services.embedding is None:
        services.errors["embedding"] = f"Unknown EMBEDDING_BACKEND {config.EMBEDDING_BACKEND!r}"
    else:
        services.embedding.set_embedding_model(
            model_id=config.EMBEDDING_MODEL_ID, embedding_size=config.EMBEDDING_MODEL_SIZE,
        )

    vdb = VectorDBProviderFactory(config=config).create(config.VECTOR_DB_BACKEND)
    if vdb is not None:
        vdb.connect()
        if getattr(vdb, "client", None) is None:
            services.errors["vector_db"] = "Could not connect to the vector database"
    services.vector_db = vdb

    services.doctor_tools = DoctorTools(
        embedding_client=services.embedding,
        vectordb_client=vdb,
        collection_name=config.VECTOR_DB_COLLECTION_NAME or "vezeeta_doctors",
    )

    if config.SKILLS_ENABLED:
        services.skills = SkillLibrary()

    if connect_mcp and config.MCP_SERVERS:
        services.mcp_registry = McpRegistry.from_env(config.MCP_SERVERS).open()

    tools = [services.doctor_tools.search_doctors]
    if config.TAVILY_API_KEY:
        tools.append(web_search)

    if services.llm is not None:
        services.agent = Agent(
            client=services.llm,
            tools=tools,
            skills=services.skills,
            mcp=services.mcp_registry,
            max_iterations=config.AGENT_MAX_STEPS,
            max_seconds=config.AGENT_MAX_SECONDS,
            max_actions=config.AGENT_MAX_ACTIONS,
            max_cost_usd=config.AGENT_MAX_COST_USD,
            max_refinements=config.AGENT_MAX_REFINEMENTS,
            verifiers=None if config.AGENT_VERIFY else [],
            input_token_cost=config.INPUT_TOKEN_COST_USD,
            output_token_cost=config.OUTPUT_TOKEN_COST_USD,
        )
    return services
