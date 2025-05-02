import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.prompts import PromptTemplate

# Load .env file
load_dotenv()

# Initialize Groq LLM
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="qwen-qwq-32b",  # or "llama3-8b-8192"
    temperature=0.7
)

# Prompt template for cover letter generation
prompt = PromptTemplate(
    input_variables=["info"],
    template="""
You are an expert career assistant who writes tailored cover letters.

Given the following information:
{info}

Write a professional, structured cover letter suitable for a job application. Use a formal tone and include relevant experience, skills, and interest in the company.
"""
)

# Tool wrapper
cover_letter_tool = Tool(
    name="CoverLetterGenerator",
    func=lambda x: (prompt | llm).invoke({"info": x}).content,
    description="Generates a professional cover letter from the user's job application details"
)

# Initialize the agent
agent = initialize_agent(
    tools=[cover_letter_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Function to run the agent
def run_cover_letter_agent(user_input: str) -> str:
    try:
        return agent.run(user_input)
    except Exception as e:
        return f"Error: {str(e)}"
