import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph

# Load environment
load_dotenv()

# Initialize Groq LLM properly
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="qwen-qwq-32b",
    temperature=0.7
)

# ------------------ TOOL 1: Cover Letter ------------------ #
cover_prompt = PromptTemplate(
    input_variables=["info"],
    template="""
You are an expert career assistant who writes tailored cover letters.

Given the following information:
{info}

Write a professional, structured cover letter suitable for a job application. Use a formal tone and include relevant experience, skills, and interest in the company.
"""
)

def generate_cover_letter(info: str) -> str:
    """Generates a cover letter based on job description and resume."""
    return (cover_prompt | llm).invoke({"info": info}).content

# ------------------ TOOL 2: Resume ------------------ #
resume_prompt = PromptTemplate(
    input_variables=["details"],
    template="""
You are a resume writing assistant.

Based on the following information provided by the user:
{details}

Create a professional resume. Organize it into the following sections:
1. Summary
2. Skills
3. Experience
4. Education
5. Projects (if any)
Use a concise and ATS-friendly format.
"""
)

def generate_resume(details: str) -> str:
    """Generates a resume based on user details."""
    return (resume_prompt | llm).invoke({"details": details}).content


from langchain_core.tools import Tool

generate_resume_tool = Tool.from_function(
    func=generate_resume,
    name="generate_resume_tool",
    description="Generate a resume from user details."
)

generate_cover_letter_tool = Tool.from_function(
    func=generate_cover_letter,
    name="generate_cover_letter_tool",
    description="Generate a cover letter from job info and resume."
)

tools = [generate_resume_tool, generate_cover_letter_tool]

# ------------------ LangGraph Agent ------------------ #
# tools = [generate_cover_letter, generate_resume]
llm_with_tools = llm.bind_tools(tools=tools)

# agent_node = create_react_agent(tools=tools, model=llm)

from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage, ToolMessage, FunctionMessage
from typing import Annotated
from langgraph.graph.message import add_messages

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from IPython.display import Image, display

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def tools_calling_llm(state:State):
    return{"messages": [llm_with_tools.invoke(state["messages"])]}

# Define state schema
class AgentState(dict): pass

graph = StateGraph(State)
graph.add_node("tools_calling_llm", tools_calling_llm)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "tools_calling_llm")
graph.add_conditional_edges("tools_calling_llm", tools_condition)
graph.add_edge("tools", "tools_calling_llm")

app_flow = graph.compile()
# display(Image(app_flow.get_graph().draw_mermaid_png()))

def run_job_agent(user_input: str) -> str:
    try:
        # Format the user input into a message for Groq API
        messages = [
            {"role": "system", "content": "You are an expert assistant for job applications."},
            {"role": "user", "content": user_input}
        ]  # Correctly formatted message list
        # print("Messages : \n", messages)
        result = app_flow.invoke({"messages": messages})  # Pass message as input
        all_msgs = result.get("messages", [])

        # ✅ Return the output of the tool (FunctionMessage)
        for msg in reversed(all_msgs):
            if isinstance(msg, FunctionMessage) and msg.content:
                return msg.content

        # ⛔ Fallback to last AIMessage if nothing else found
        for msg in reversed(all_msgs):
            if hasattr(msg, "content") and msg.content:
                return msg.content

        return "No usable output from agent."
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    question="make a resume"
    response = app_flow.invoke({"messages": question})
    # print(response)
    for m in response['messages']:
        m.pretty_print()
