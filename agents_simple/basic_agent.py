from crewai import Agent, Crew, Task, LLM
from dotenv import load_dotenv
import os

# groq api error - it doesnt have this field in its api request format
import crewai.llms.cache as crew_cache
crew_cache.mark_cache_breakpoint = lambda msg: msg

load_dotenv()  # Load environment variables from .env file


GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# config the LLM to be used by the agent
myLLM = LLM(
    model="groq/qwen/qwen3.8-27b",      # use model provider in the name as well
    api_key=GROQ_API_KEY,
    temperature=0.7,
    max_tokens=512,
    verbose=True,
)

# print("output: ", myLLM.call("Write a short poem about AI agents."))

# create an agent with LLM
email_assistant = Agent(
    role="Email Assistant Agent",
    goal="An AI agent that helps you write and manage emails.",
    backstory="A highly experienced communication expert skilled in professional email writing",
    verbose=True,
    llm=myLLM,
)


original_email = """
hey team, just wanted to tell u that the demo is kind of ready, but there's still stuff left.
Maybe we can show what we have and say rest is WIP.
Let me know what u think. thanks
"""

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