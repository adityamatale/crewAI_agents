from crewai import Agent, Crew, LLM, Task
from crewai_tools import SerperDevTool
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv


load_dotenv()  # Load environment variables from .env file


@CrewBase
class BlogWriterCrew():
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['research_agent'], # type: ignore[index]
            tools=[SerperDevTool()],
            verbose=True
        )

    @agent
    def writer(self) -> Agent:
        return Agent(
            config=self.agents_config['writer_agent'], # type: ignore[index]
            verbose=True
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'], # type: ignore[index]
            agent = self.researcher()
        )

    @task
    def blog_task(self) -> Task:
        return Task(
            config=self.tasks_config['blog_task'], # type: ignore[index]
            agent = self.writer()
        )

    @crew
    def blog_writer_crew(self) -> Crew:
        return Crew(
            agents=[self.researcher(), self.writer()],
            tasks=[self.research_task(), self.blog_task()],
            verbose=True,
        )

if __name__ == "__main__":
    crew_instance = BlogWriterCrew()
    crew_instance.blog_writer_crew().kickoff(inputs={"topic": "The impact of AI on modern education"})