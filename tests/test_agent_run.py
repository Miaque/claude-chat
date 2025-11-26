import uuid

from models.agent_run import AgentRuns


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
