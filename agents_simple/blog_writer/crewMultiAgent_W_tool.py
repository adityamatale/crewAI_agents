from crewai import Agent, Crew, LLM, Task
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# create llm to use
myLLM = LLM(
    model='openai/gpt-6-luna',
    # temperature=0.1,
)

#create agents to use
research_agent = Agent(
    role="Research Specialist",
    goal="Research interesting facts about the topic: {topic}",
    backstory="You are an expert at finding relevant and factual data.",
    verbose=True,
    tools=[SerperDevTool()],
    llm=myLLM,
    max_iter=2
)

writer_agent = Agent(
    role="Creative Writer",
    goal="Write a short blog summary using the research",
    backstory="You are skilled at writing engaging summaries based on provided content.",
    llm=myLLM,
    verbose=True,
)


# create tasks for the agents
task1 = Task(
    description="Find 3-5 interesting and recent facts about {topic}.",
    expected_output="A bullet list of 3-5 facts",
    agent=research_agent,
)

# task 2 has context of task 1, so it can use the output of task 1 to write the blog summary
task2 = Task(
    description="Write a 100-word blog post summary about {topic} using the facts from the research.",
    expected_output="A blog post summary",
    agent=writer_agent,
    context=[task1],
)

# create a crew with the agents and tasks
crew = Crew(
    agents=[research_agent, writer_agent],
    tasks=[task1, task2],
    verbose=True,
)


# here we use the inputs parameter to update the prompts dynamically with the topic we want to research and write about
crew.kickoff(inputs = {"topic":"The impact of AI on modern education"})