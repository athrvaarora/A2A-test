import os
from typing import Dict, Any, AsyncIterable, List
from uuid import uuid4

from crewai import Agent, Task, Crew
from langchain_google_genai import ChatGoogleGenerativeAI


class AnalysisAgent:
    """Agent that can analyze problems, data, or situations and provide insights."""
    
    def __init__(self, api_key=None):
        """Initialize the analysis agent."""
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            
        self.model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
        
        # Define the analysis agents in our crew
        self.data_analyzer = Agent(
            role="Data Analyzer",
            goal="Thoroughly analyze the problem or situation to identify key patterns and insights",
            backstory="""You are an expert at breaking down complex problems and
            analyzing situations methodically. You have a strong background in
            data analysis, critical thinking, and problem identification.""",
            verbose=True,
            allow_delegation=False,
            llm=self.model
        )
        
        self.insight_generator = Agent(
            role="Insight Generator",
            goal="Generate valuable insights and recommendations based on analysis",
            backstory="""You excel at drawing meaningful conclusions from 
            analysis and providing actionable insights. You can synthesize
            complex information into clear, valuable recommendations.""",
            verbose=True,
            allow_delegation=True,
            llm=self.model
        )
        
        self.critic = Agent(
            role="Analysis Critic",
            goal="Evaluate the analysis and insights for rigor, completeness, and validity",
            backstory="""You have exceptional critical thinking skills and can
            identify gaps, biases, or weaknesses in analysis. You ensure that
            conclusions are well-supported by evidence and logic.""",
            verbose=True,
            allow_delegation=True,
            llm=self.model
        )
    
    def _create_crew_for_query(self, query: str, session_id: str) -> Crew:
        """Create a crew with tasks based on the query."""
        task1 = Task(
            description=f"Analyze this problem/situation in depth: {query}",
            agent=self.data_analyzer,
            expected_output="A thorough analysis identifying key factors, patterns, and potential issues",
            context=f"The user requested analysis of: {query}"
        )
        
        task2 = Task(
            description="Generate insights and recommendations based on the analysis",
            agent=self.insight_generator,
            expected_output="A set of valuable insights and actionable recommendations",
            context="Build upon the analysis to provide meaningful conclusions",
            dependencies=[task1]
        )
        
        task3 = Task(
            description="Critique and refine the analysis and insights",
            agent=self.critic,
            expected_output="A refined analysis with validated insights and recommendations",
            context="Evaluate the strength of the analysis and insights, addressing any weaknesses",
            dependencies=[task2]
        )
        
        crew = Crew(
            agents=[self.data_analyzer, self.insight_generator, self.critic],
            tasks=[task1, task2, task3],
            verbose=True,
            process=Crew.Process.SEQUENTIAL,
            manager_llm=self.model,
            session_id=session_id
        )
        
        return crew
    
    def invoke(self, query: str, session_id: str = None) -> Dict[str, Any]:
        """Process a query and return analysis results."""
        if not session_id:
            session_id = uuid4().hex
            
        crew = self._create_crew_for_query(query, session_id)
        
        # Run the crew to generate analysis
        result = crew.kickoff()
        
        return {
            "is_task_complete": True,
            "require_user_input": False,
            "content": result,
        }
    
    async def stream(self, query: str, session_id: str = None) -> AsyncIterable[Dict[str, Any]]:
        """Process a query and stream the analysis process."""
        if not session_id:
            session_id = uuid4().hex
            
        crew = self._create_crew_for_query(query, session_id)
        
        # Simulate streaming by yielding updates for each step
        yield {
            "is_task_complete": False,
            "require_user_input": False,
            "content": "Analyzing the problem/situation in depth...",
        }
        
        # Run the first task
        task1_result = crew.tasks[0].execute()
        
        yield {
            "is_task_complete": False,
            "require_user_input": False,
            "content": "Analysis complete! Now generating insights and recommendations...",
        }
        
        # Update context for task 2
        crew.tasks[1].context = f"Analysis results: {task1_result}. Generate insights based on this."
        task2_result = crew.tasks[1].execute()
        
        yield {
            "is_task_complete": False,
            "require_user_input": False,
            "content": "Insights generated! Now evaluating and refining the analysis...",
        }
        
        # Update context for task 3
        crew.tasks[2].context = f"Analysis and insights to review: {task2_result}"
        final_result = crew.tasks[2].execute()
        
        yield {
            "is_task_complete": True,
            "require_user_input": False,
            "content": final_result,
        }
    
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"] 