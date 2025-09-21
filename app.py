import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up Streamlit page configuration for a light, modern look
st.set_page_config(
    page_title="Mental Wellness Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern light theme
st.markdown("""
    <style>
    /* General body styling */
    body {
        background-color: #f0f4f8;
        color: #333333;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Header styling */
    .stApp > header {
        background-color: #ffffff;
    }
    
    /* Sidebar styling */
    .css-1lcbmhc {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
    
    /* Chat message containers */
    .st-chat-message {
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 12px;
    }
    
    /* User messages */
    .st-chat-message.user {
        background-color: #e3f2fd;
        color: #1565c0;
    }
    
    /* Assistant messages */
    .st-chat-message.assistant {
        background-color: #ffffff;
        color: #333333;
        border: 1px solid #e0e0e0;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #4caf50;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 16px;
    }
    
    .stButton > button:hover {
        background-color: #388e3c;
    }
    
    /* Info boxes */
    .st-info {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        padding: 12px;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Mental health disclaimer text
st.title("🧠 Mental Wellness Chatbot")
st.info("""
**Important Note on Mental Health:**  
This chatbot is designed to provide general support, empathy, and coping strategies based on common mental wellness practices. It is not a substitute for professional medical advice, diagnosis, or treatment. If you are experiencing a mental health crisis, please seek help from a qualified professional or contact emergency services immediately.  

Some helpful resources:  
- National Suicide Prevention Lifeline (US): 988  
- Crisis Text Line: Text HOME to 741741  
- For international help: Visit https://www.befrienders.org  

Remember, taking care of your mental health is important. Practice self-compassion, mindfulness, and reach out to loved ones when needed.
""")

# Sidebar with predefined prompts
st.sidebar.title("Quick Start Prompts")
predefined_prompts = [
    "I'm feeling anxious about work. What can I do?",
    "How can I practice mindfulness daily?",
    "I'm having trouble sleeping. Any tips?",
    "What are some ways to build self-esteem?",
    "I feel overwhelmed. Help me prioritize."
]

for prompt in predefined_prompts:
    if st.sidebar.button(prompt):
        st.session_state.user_input = prompt

# Retrieve Groq API key from environment variables
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ_API_KEY not found in .env file. Please set it and try again.")
    st.stop()

# System prompt for mental wellness
system_prompt = """
You are a compassionate and supportive mental wellness assistant. Your goal is to provide empathetic, non-judgmental responses. Offer practical coping strategies, mindfulness exercises, or general advice based on established mental health practices. Always remind the user that you are not a licensed therapist and encourage seeking professional help if issues are serious. Keep responses positive, encouraging, and focused on empowerment. Do not diagnose or prescribe treatments.
"""

# Set up LangChain with Groq
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile",  # Fallback to a supported model since llama-3.1-70b-versatile is decommissioned
    temperature=0.7
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

memory = ConversationBufferMemory(memory_key="chat_history")
chain = LLMChain(llm=llm, prompt=prompt_template, memory=memory)

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_input = st.chat_input("How are you feeling today?")

if "user_input" in st.session_state:
    user_input = st.session_state.user_input
    del st.session_state.user_input

if user_input:
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chain.run(input=user_input)
        st.markdown(response)
    
    # Add assistant message to chat
    st.session_state.messages.append({"role": "assistant", "content": response})