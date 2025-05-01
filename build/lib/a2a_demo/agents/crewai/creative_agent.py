import os
from typing import Dict, Any, AsyncIterable, List
from uuid import uuid4

from crewai import Agent, Task, Crew
from crewai.tasks.task_output import TaskOutput
from langchain_google_genai import ChatGoogleGenerativeAI


class CreativeAgent:
    """Agent that can generate creative content like stories, ideas, and marketing materials."""
    
    def __init__(self, api_key=None):
        """Initialize the creative agent."""
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            
        self.model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
        
        # Define the creative agents in our crew
        self.idea_generator = Agent(
            role="Idea Generator",
            goal="Generate innovative, original ideas based on the given brief",
            backstory="""You are a highly creative thinker with a knack for coming up 
            with unique, outside-the-box ideas. You have a background in brainstorming 
            and concept development.""",
            verbose=True,
            allow_delegation=False,
            llm=self.model
        )
        
        self.content_creator = Agent(
            role="Content Creator",
            goal="Transform ideas into polished, engaging content",
            backstory="""You are a skilled writer and content developer who can take
            raw ideas and turn them into compelling narratives, marketing copy, or
            other creative content formats.""",
            verbose=True,
            allow_delegation=True,
            llm=self.model
        )
        
        self.critic = Agent(
            role="Creative Critic",
            goal="Evaluate and refine creative work to ensure quality and effectiveness",
            backstory="""You have a keen eye for detail and a deep understanding of what
            makes content engaging and effective. You can identify weaknesses and suggest
            improvements to any creative work.""",
            verbose=True,
            allow_delegation=True,
            llm=self.model
        )
    
    def _create_crew_for_query(self, query: str, session_id: str) -> Crew:
        """Create a crew with tasks based on the query."""
        task1 = Task(
            description=f"Generate creative ideas based on this brief: {query}",
            agent=self.idea_generator,
            expected_output="A list of 3-5 creative ideas with brief explanations",
            context=f"The user requested: {query}"
        )
        
        task2 = Task(
            description="Develop the best idea into full content",
            agent=self.content_creator,
            expected_output="Fully developed creative content based on the chosen idea",
            context="Use the output from the idea generation phase and develop the most promising concept",
            dependencies=[task1]
        )
        
        task3 = Task(
            description="Review and refine the content",
            agent=self.critic,
            expected_output="Final polished content with improvements",
            context="Critique the content for quality, engagement, and effectiveness",
            dependencies=[task2]
        )
        
        crew = Crew(
            agents=[self.idea_generator, self.content_creator, self.critic],
            tasks=[task1, task2, task3],
            verbose=True,
            process=Crew.Process.SEQUENTIAL,
            manager_llm=self.model,
            session_id=session_id
        )
        
        return crew
    
    def invoke(self, query: str, session_id: str = None) -> Dict[str, Any]:
        """Process a query and return creative content."""
        if not session_id:
            session_id = uuid4().hex
            
        crew = self._create_crew_for_query(query, session_id)
        
        # Run the crew to generate content
        result = crew.kickoff()
        
        return {
            "is_task_complete": True,
            "require_user_input": False,
            "content": result,
        }
    
    async def stream(self, query: str, session_id: str = None) -> AsyncIterable[Dict[str, Any]]:
        """Process a query and stream the creative process."""
        if not session_id:
            session_id = uuid4().hex
            
        crew = self._create_crew_for_query(query, session_id)
        
        # Simulate streaming by yielding updates for each step
        yield {
            "is_task_complete": False,
            "require_user_input": False,
            "content": "Generating creative ideas based on your request...",
        }
        
        # Run the first task
        task1_result = crew.tasks[0].execute()
        
        yield {
            "is_task_complete": False,
            "require_user_input": False,
            "content": "Ideas generated! Now developing the best idea into full content...",
        }
        
        # Update context for task 2
        crew.tasks[1].context = f"Ideas generated: {task1_result}. Develop the most promising one."
        task2_result = crew.tasks[1].execute()
        
        yield {
            "is_task_complete": False,
            "require_user_input": False,
            "content": "Content created! Now refining and polishing the work...",
        }
        
        # Update context for task 3
        crew.tasks[2].context = f"Content to review: {task2_result}"
        final_result = crew.tasks[2].execute()
        
        yield {
            "is_task_complete": True,
            "require_user_input": False,
            "content": final_result,
        }
    
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"] 