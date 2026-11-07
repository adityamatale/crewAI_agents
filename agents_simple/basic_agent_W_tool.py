from crewai import Agent, Crew, Task, LLM
from crewai.tools import BaseTool
from dotenv import load_dotenv
import os

# # groq api error - it doesnt have this field in its api request format
# import crewai.llms.cache as crew_cache
# crew_cache.mark_cache_breakpoint = lambda msg: msg

load_dotenv()  # Load environment variables from .env file


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# config the LLM to be used by the agent
myLLM = LLM(
    model="openai/gpt-6-luna",      # use model provider in the name as well
    api_key=OPENAI_API_KEY,
    temperature=0.7,
)

# print("output: ", myLLM.call("Write a short poem about AI agents."))


#create a tool for the agent to use
class ReplaceJargonsTool(BaseTool):
    name: str = "Jargon replacement tool"
    description : str = "Replaces jargon with more specific terms. "

    def _run(self, email: str) -> str:
        replacements = {
            "PRX": "Project Phoenix (internal AI revamp project)",
            "TAS": "technical architecture stack",
            "DBX": "client database cluster",
            "SDS": "Smart Data Syncer",
            "SYNCBOT": "internal standup assistant bot",
            "WIP": "in progress",
            "POC": "proof of concept",
            "ping": "reach out"
        }
        suggestions = []
        email_lower = email.lower()
        for jargon, replacement in replacements.items():
            if jargon.lower() in email_lower:
                suggestions.append(f"Consider replacing '{jargon}' with '{replacement}'")

        return "\n".join(suggestions) if suggestions else "No jargon or internal abbreviations detected."


jt = ReplaceJargonsTool()

# create an agent with LLM
email_assistant = Agent(
    role="Email Assistant Agent",
    goal="An AI agent that helps you write and manage emails.",
    backstory="A highly experienced communication expert skilled in professional email writing",
    verbose=True,
    tools=[jt],
    llm=myLLM,
)


# original_email = """
# hey team, just wanted to tell u that the demo is kind of ready, but there's still stuff left.
# Maybe we can show what we have and say rest is WIP.
# Let me know what u think. thanks
# """



original_email = """
looping in Priya. TAS and PRX updates are in the deck. ETA for SDS integration is Friday.
Let's sync up tomorrow if SYNCBOT allows 😄. ping me if any blockers.
"""

# print(jt.run(original_email))

# exit()

# create a task for an agent/ or that some agent can do and specify the agent
email_task = Task(
    description=f"""Take the following rough email and rewrite it into a professional and polished version.
    Expand abbreviations:
    '''{original_email}'''""",
    agent=email_assistant,
    expected_output="A professional and polished version of the original email, with expanded abbreviations and improved clarity.",
)


# finally create the crew | this is the most imp part as its part of the framework
myCREW = Crew(
    agents=[email_assistant],
    tasks=[email_task],
    verbose=True,
)


# call the crew to run the tasks
print(myCREW.kickoff())