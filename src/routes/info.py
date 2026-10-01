"""Introspection endpoints: health, skills, tools."""

from fastapi import APIRouter, HTTPException, Request

router = APIRouter(tags=["Info"])


@router.get("/health")
async def health(req: Request):
    services = getattr(req.app.state, "services", None)
    if services is None:
        return {"status": "starting"}
    vdb_ok = getattr(services.vector_db, "client", None) is not None
    ok = services.agent is not None and vdb_ok
    return {
        "status": "ok" if ok else "degraded",
        "llm": {"backend": services.config.GENERATION_BACKEND, "model": services.config.GENERATION_MODEL_ID,
                "ready": services.agent is not None},
        "vector_db": {"backend": services.config.VECTOR_DB_BACKEND, "connected": vdb_ok,
                      "mode": "cloud" if services.config.VECTOR_DB_URL else "local"},
        "skills": len(services.skills.list()) if services.skills else 0,
        "mcp_server": req.app.state.mcp_server is not None,
        "mcp_clients": services.mcp_registry.status() if services.mcp_registry else {},
        "telegram": getattr(req.app.state, "tg_handler", None) is not None,
        "active_sessions": len(req.app.state.sessions),
        "errors": services.errors,
    }


@router.get("/skills")
async def list_skills(req: Request):
    skills = getattr(req.app.state, "skills", None)
    return [s.as_dict() for s in skills.list()] if skills else []


@router.get("/skills/{name}")
async def get_skill(name: str, req: Request):
    skills = getattr(req.app.state, "skills", None)
    skill = skills.get(name) if skills else None
    if not skill:
        raise HTTPException(status_code=404, detail=f"No skill named {name!r}")
    return {**skill.as_dict(), "instructions": skill.body()}


@router.get("/tools")
async def list_tools(req: Request):
    agent = getattr(req.app.state, "agent", None)
    if agent is None:
        return {"tools": []}
    budget = agent.new_budget()
    return {
        "tools": agent.tool_names(),
        "budget_per_message": {
            "max_steps": budget.max_steps, "max_seconds": budget.max_seconds,
            "max_actions": budget.max_actions, "max_cost_usd": budget.max_cost_usd,
        },
        "verifiers": [v.__name__ for v in agent.verifiers],
    }
