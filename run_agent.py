"""Build Agent using Microsoft Agent Framework in Python
# Run this python script
> pip install agent-framework==1.0.0rc3 agent-framework-azure-ai==1.0.0rc3
> python <this-script-path>.py
"""

import asyncio
import logging
import os
from dotenv import load_dotenv
from azure.core.pipeline.transport._aiohttp import AioHttpTransport as _AioHttpTransport
_orig_open = _AioHttpTransport.open

async def _patched_open(self):
    if not self.session and self._session_owner:
        import aiohttp
        self.session = aiohttp.ClientSession(
            trust_env=self._use_env_settings,
            cookie_jar=aiohttp.DummyCookieJar(),
            auto_decompress=True,
        )
        self._has_been_opened = True
        await self.session.__aenter__()
    else:
        await _orig_open(self)

_AioHttpTransport.open = _patched_open

from agent_framework import Agent
from agent_framework.azure import AzureAIClient
from azure.ai.projects.aio import AIProjectClient
from azure.identity.aio import DefaultAzureCredential

load_dotenv()

# User inputs for the conversation
USER_INPUTS = [
    "If we onboard a US vendor after Schrems II, what safeguards should we implement for personal data transfers?",
]

async def main() -> None:
    async with (
        # For authentication, DefaultAzureCredential supports multiple authentication methods. Run `az login` in terminal for Azure CLI auth.
        DefaultAzureCredential() as credential,
        AIProjectClient(endpoint=os.environ["AZURE_AI_PROJECT_ENDPOINT"], credential=credential) as project_client,
        Agent(
            client=AzureAIClient(
                project_client=project_client,
                agent_name="ComplianceAgent",
                model_deployment_name="gpt-4o",
                use_latest_version=True, # re-use latest agent version instead of creating one
            ),
        ) as agent,
    ):
        # Process user messages
        for user_input in USER_INPUTS:
            print(f"\n# User: '{user_input}'")
            printed_tool_calls = set()
            async for chunk in agent.run([user_input], stream=True):
                # log tool calls if any
                function_calls = [
                    c for c in chunk.contents
                    if c.type == "function_call"
                ]
                for call in function_calls:
                    if call.call_id not in printed_tool_calls:
                        print(f"Tool calls: {call.name}")
                        printed_tool_calls.add(call.call_id)
                if chunk.text:
                    print(chunk.text, end="")
            print("")
        
        print("\n--- All tasks completed successfully ---")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Program finished.")
