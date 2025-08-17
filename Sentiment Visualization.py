# pip install llama-index llama-index-core llama-index-embeddings-huggingface llama-index-memory-mem0 llama-index-llms-groq

from llama_index.llms.groq import Groq
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding 
from llama_index.memory.mem0 import Mem0Memory
from langchain_groq import ChatGroq
import os 

from dotenv import load_dotenv
load_dotenv()

os.environ["MeM0_API_KEY"] = os.getenv("MEM0_API_KEY")
API_KEY = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = API_KEY

import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

MODEL_NAME = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5" 
KNOWLEDGE_BASE = "./budget_data/"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 20
OUTPUT_TOKENS = 512 

def get_llm(model_name, api_key):
    return Groq(model=model_name, api_key=api_key, temperature=0.5)

os.environ["GROQ_API_KEY"]   = API_KEY
llm = ChatGroq(model="llama3-70b-8192", temperature=0.8)

# Custom function to analyze feedback
def comment_on_feedback(feedback: str) -> str:
    prompt = f"""
    Analyze this customer feedback, decide only whether it is positive, negative, or neutral, no description.

    Feedback: "{feedback}"
    """
    return llm.invoke(prompt).content

def initialize_settings():
    Settings.llm = get_llm(MODEL_NAME, API_KEY)
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
    Settings.num_output = OUTPUT_TOKENS 
    Settings.node_parser = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP) 

context = {"user_id": "test_user_1"}
memory_from_client = Mem0Memory.from_client(
    context=context,
    api_key=os.environ["MeM0_API_KEY"],
    search_msg_limit=50  
)

feedback_log = [
    {"timestamp": datetime(2025, 8, 10).date(), "sentiment": "Positive"},
    {"timestamp": datetime(2025, 8, 10).date(), "sentiment": "Positive"},
    {"timestamp": datetime(2025, 8, 10).date(), "sentiment": "Positive"},
    {"timestamp": datetime(2025, 8, 10).date(), "sentiment": "Negative"},
    {"timestamp": datetime(2025, 8, 10).date(), "sentiment": "Neutral"},
    {"timestamp": datetime(2025, 8, 10).date(), "sentiment": "Negative"},
    {"timestamp": datetime(2025, 8, 10).date(), "sentiment": "Positive"},
    {"timestamp": datetime(2025, 8, 10).date(), "sentiment": "Positive"},
    {"timestamp": datetime(2025, 8, 10).date(), "sentiment": "Negative"},
    {"timestamp": datetime(2025, 8, 10).date(), "sentiment": "Negative"},
    {"timestamp": datetime(2025, 8, 10).date(), "sentiment": "Neutral"},

    {"timestamp": datetime(2025, 8, 11).date(), "sentiment": "Positive"},
    {"timestamp": datetime(2025, 8, 11).date(), "sentiment": "Negative"},
    {"timestamp": datetime(2025, 8, 11).date(), "sentiment": "Negative"},
    {"timestamp": datetime(2025, 8, 11).date(), "sentiment": "Neutral"},
    {"timestamp": datetime(2025, 8, 11).date(), "sentiment": "Positive"},
    {"timestamp": datetime(2025, 8, 11).date(), "sentiment": "Positive"},
    {"timestamp": datetime(2025, 8, 11).date(), "sentiment": "Negative"},
    {"timestamp": datetime(2025, 8, 11).date(), "sentiment": "Neutral"},
    {"timestamp": datetime(2025, 8, 11).date(), "sentiment": "Negative"},
    {"timestamp": datetime(2025, 8, 11).date(), "sentiment": "Positive"},
    {"timestamp": datetime(2025, 8, 11).date(), "sentiment": "Positive"},

    {"timestamp": datetime(2025, 8, 12).date(), "sentiment": "Positive"},
    {"timestamp": datetime(2025, 8, 12).date(), "sentiment": "Positive"},
    {"timestamp": datetime(2025, 8, 12).date(), "sentiment": "Neutral"},
    {"timestamp": datetime(2025, 8, 12).date(), "sentiment": "Negative"},
    {"timestamp": datetime(2025, 8, 12).date(), "sentiment": "Positive"},
    {"timestamp": datetime(2025, 8, 12).date(), "sentiment": "Positive"},
    {"timestamp": datetime(2025, 8, 12).date(), "sentiment": "Negative"},
    {"timestamp": datetime(2025, 8, 12).date(), "sentiment": "Negative"},
    {"timestamp": datetime(2025, 8, 12).date(), "sentiment": "Neutral"},
    {"timestamp": datetime(2025, 8, 12).date(), "sentiment": "Positive"},

    {"timestamp": datetime(2025, 8, 13).date(), "sentiment": "Negative"},
    {"timestamp": datetime(2025, 8, 13).date(), "sentiment": "Negative"},
    {"timestamp": datetime(2025, 8, 13).date(), "sentiment": "Neutral"},
    {"timestamp": datetime(2025, 8, 13).date(), "sentiment": "Positive"},
    {"timestamp": datetime(2025, 8, 13).date(), "sentiment": "Positive"},

    {"timestamp": datetime(2025, 8, 14).date(), "sentiment": "Positive"},
    {"timestamp": datetime(2025, 8, 14).date(), "sentiment": "Neutral"},
    {"timestamp": datetime(2025, 8, 14).date(), "sentiment": "Negative"},
    {"timestamp": datetime(2025, 8, 14).date(), "sentiment": "Positive"},
    {"timestamp": datetime(2025, 8, 14).date(), "sentiment": "Positive"},

    {"timestamp": datetime(2025, 8, 15).date(), "sentiment": "Negative"},
    {"timestamp": datetime(2025, 8, 15).date(), "sentiment": "Neutral"},
    {"timestamp": datetime(2025, 8, 15).date(), "sentiment": "Positive"},
    {"timestamp": datetime(2025, 8, 15).date(), "sentiment": "Positive"},
    {"timestamp": datetime(2025, 8, 15).date(), "sentiment": "Positive"},
    {"timestamp": datetime(2025, 8, 15).date(), "sentiment": "Positive"},
    {"timestamp": datetime(2025, 8, 15).date(), "sentiment": "Neutral"},
    {"timestamp": datetime(2025, 8, 15).date(), "sentiment": "Negative"},
    {"timestamp": datetime(2025, 8, 15).date(), "sentiment": "Positive"},
    {"timestamp": datetime(2025, 8, 15).date(), "sentiment": "Positive"},

    {"timestamp": datetime(2025, 8, 16).date(), "sentiment": "Neutral"},
    {"timestamp": datetime(2025, 8, 16).date(), "sentiment": "Negative"},
    {"timestamp": datetime(2025, 8, 16).date(), "sentiment": "Positive"},
    {"timestamp": datetime(2025, 8, 16).date(), "sentiment": "Positive"},
    {"timestamp": datetime(2025, 8, 16).date(), "sentiment": "Neutral"},
    {"timestamp": datetime(2025, 8, 16).date(), "sentiment": "Positive"},
    {"timestamp": datetime(2025, 8, 16).date(), "sentiment": "Neutral"},
    {"timestamp": datetime(2025, 8, 16).date(), "sentiment": "Negative"},
    {"timestamp": datetime(2025, 8, 16).date(), "sentiment": "Positive"},
    {"timestamp": datetime(2025, 8, 16).date(), "sentiment": "Positive"},
]
   
user_input = input("Enter your feedback here: ")

result = comment_on_feedback(user_input)
print(result)

def log_feedback(sentiment, feedback_date = None):
    feedback_log.append({
        "timestamp": feedback_date or datetime.now().date(),
        "sentiment": sentiment
    })

new_sentiment = comment_on_feedback(user_input).strip()
log_feedback(new_sentiment)

df_feedback = pd.DataFrame(feedback_log)
df_plot = df_feedback.groupby([df_feedback['timestamp'], 'sentiment']).size().unstack(fill_value=0)


end_date = datetime.now().date()
start_date = end_date - timedelta(days=7)
all_dates = pd.date_range(start=start_date, end=end_date)
df_filtered = df_plot.reindex(all_dates, fill_value=0)

df_filtered.plot(kind='line', marker='o', figsize=(10,5))
plt.title("Feedback Sentiments Over Time")
plt.xlabel("Date")
plt.ylabel("Count")
plt.grid(True)
plt.show()