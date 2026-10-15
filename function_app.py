from pathlib import Path

import azure_functions_agents.app as agent_app
import azure_functions_agents.discovery.tools as tool_discovery
from azure_functions_agents import create_function_app
from azure_functions_agents.discovery.tools import ProjectTools
import orchestrators.modular_orchestrator  # noqa: F401  attaches tournament functions to the shared blueprint
from orchestrators.loop_optimizer import bp as loop_bp


def _discover(_app_root: Path):
    return ProjectTools(user_tools=[], workflow_tools=[], failed_loads=[])


tool_discovery.discover_project_tools = _discover
agent_app.discover_project_tools = _discover

app = create_function_app(app_root=Path(__file__).resolve().parent)
app.register_blueprint(loop_bp)
