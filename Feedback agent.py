import os
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

os.environ["GROQ_API_KEY"]   = GROQ_API_KEY

from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool 
from langchain_groq import ChatGroq

llm = ChatGroq(model="llama3-70b-8192", temperature=0.8)

VERBOSE = False

# Custom function to analyze feedback
def comment_on_feedback(feedback: str) -> str:
    prompt = f"""
    Analyze this customer feedback, decide if it is positive, negative, or neutral, and give a short explanation.

    Feedback: "{feedback}"
    """
    return llm.invoke(prompt).content

def reply_to_customer(reply: str) -> str:
    prompt = f"""
    Write a polite, creative, and professional reply to the following feedback about 2 to 3 sentences:

    Feedback: "{reply}"
    """
    return llm.invoke(prompt).content

# Wrap it in a LangChain Tool
feedback_tool = Tool(
    name="FeedbackCommenterOnSteamNoodles",
    func=comment_on_feedback,
    description="Analyze customer feedback and summarize it as positive, negative, or neutral with reasoning."
)

replying_tool = Tool(
    name="ReplyToTheCustomer",
    func=reply_to_customer,
    description="Write a polite, friendly, and creative reply to the customer based on their feedback."
)

# Initialize Agent with the tool
agent = initialize_agent(
    tools=[feedback_tool, replying_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=VERBOSE
)

user_input = input("Welcome to SteamNoodles, Enter your feedback here: ")

analyzer = comment_on_feedback(user_input)
comment = reply_to_customer(user_input)

print(analyzer)
print(comment)