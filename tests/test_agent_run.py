import uuid

from core.services.db import get_db
from models.agent_run import AgentRun, AgentRuns


def test_get_agent_run_by_id():
    agent_run_id = str(uuid.uuid4())
    agent_run = AgentRuns.get_by_id(agent_run_id)
    print(agent_run)
    print("\n")

    agent_run_id = "1a90e93f-f802-4519-8645-c1f983c84e77"
    agent_run = AgentRuns.get_by_id(agent_run_id)
    print(agent_run)
    print("\n")

    agent_run_data = agent_run.model_dump() if agent_run else None
    print(agent_run_data)
    print("\n")

    agent_run_data = agent_run.model_dump(mode="json") if agent_run else None
    print(agent_run_data)
    print("\n")

    agent_run_data = agent_run.model_dump_json() if agent_run else None
    print(agent_run_data)


def test_update_agent_run_status():
    # with get_db() as db:
    #     verify_result = (
    #         db.query(AgentRun)
    #         .filter(AgentRun.id == "1a90e93f-f802-4519-8645-c1f983c84e77")
    #         .with_entities(
    #             AgentRun.status.label("status"),
    #             AgentRun.completed_at.label("completed_at"),
    #         )
    #         .first()
    #     )

    with get_db() as db:
        verify_result = (
            db.query(AgentRun.status, AgentRun.completed_at)
            .filter(AgentRun.id == "1a90e93f-f802-4519-8645-c1f983c84e77")
            .first()
        )

    print(verify_result)
    # 现在你可以通过 verify_result.status 访问状态
    if verify_result:
        print(f"Status: {verify_result.status}")
        print(f"Completed at: {verify_result.completed_at}")
