import streamlit as st
import uuid
import json

from langchain_core.messages import HumanMessage, AIMessage
from agent_backend import chatbot


# ----------------------------- Utilities -----------------------------
def generate_thread_id():
    return str(uuid.uuid4())

def add_new_thread(thread_id: str):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_new_thread(thread_id)
    st.session_state["message_history"] = []
    st.session_state["ehr_json"] = {}
    st.session_state["user_profile"] = default_user_profile()
    st.session_state["hard_constraints"] = []

def load_conversation(thread_id: str):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])

def conversation_title(thread_id: str):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    messages = state.values.get("messages", [])
    for msg in messages:
        if msg.type == "human":
            return (msg.content[:60] + " ...") if len(msg.content) > 60 else msg.content
    return "Current Conversation"

def default_user_profile():
    return {
        "age": None,
        "sex": None,
        "height_cm": None,
        "weight_kg": None,
        "activity_level": "moderate",
        "dietary_pattern": "mixed",
        "culture_or_cuisine": "Bangladeshi",
        "country": "Bangladesh",
        "manual_location": "Dhaka",  # you can set None to use IP geo
        "hemisphere": "north",
        "budget_level": "medium",  # low/medium/high
    }


# ----------------------------- Session State -----------------------------
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "ehr_json" not in st.session_state:
    st.session_state["ehr_json"] = {}

if "user_profile" not in st.session_state:
    st.session_state["user_profile"] = default_user_profile()

if "hard_constraints" not in st.session_state:
    st.session_state["hard_constraints"] = []

add_new_thread(st.session_state["thread_id"])


# ----------------------------- Sidebar -----------------------------
st.sidebar.title("🥗 Personalized Nutrition AI Agent")

if st.sidebar.button("➕ New Chat"):
    reset_chat()

st.sidebar.header("User Profile")
with st.sidebar.expander("Edit profile", expanded=True):
    prof = st.session_state["user_profile"]

    prof["age"] = st.number_input("Age", min_value=0, max_value=120, value=prof["age"] or 0)
    prof["sex"] = st.selectbox("Sex", ["unknown", "female", "male"], index=["unknown","female","male"].index(prof["sex"] or "unknown"))
    prof["height_cm"] = st.number_input("Height (cm)", min_value=50, max_value=250, value=int(prof["height_cm"] or 165))
    prof["weight_kg"] = st.number_input("Weight (kg)", min_value=20, max_value=300, value=int(prof["weight_kg"] or 65))

    prof["activity_level"] = st.selectbox("Activity level", ["sedentary", "light", "moderate", "active"], index=["sedentary","light","moderate","active"].index(prof.get("activity_level","moderate")))
    prof["dietary_pattern"] = st.selectbox("Diet pattern", ["mixed", "vegetarian", "vegan", "halal", "keto", "low_carb"], index=["mixed","vegetarian","vegan","halal","keto","low_carb"].index(prof.get("dietary_pattern","mixed")))
    prof["culture_or_cuisine"] = st.text_input("Cuisine preference", value=prof.get("culture_or_cuisine","Bangladeshi"))

    prof["country"] = st.text_input("Country", value=prof.get("country","Bangladesh"))
    prof["manual_location"] = st.text_input("City/Location (optional)", value=prof.get("manual_location","Dhaka"))
    prof["budget_level"] = st.selectbox("Budget level", ["low", "medium", "high"], index=["low","medium","high"].index(prof.get("budget_level","medium")))

    st.session_state["user_profile"] = prof

st.sidebar.header("Hard Constraints")
constraints_text = st.sidebar.text_area(
    "Add constraints (one per line). Examples:\n- no alcohol\n- no grapefruit\n- low potassium\n- low sodium",
    value="\n".join(st.session_state["hard_constraints"]),
    height=120,
)
st.session_state["hard_constraints"] = [c.strip() for c in constraints_text.splitlines() if c.strip()]

st.sidebar.header("EHR Upload (JSON)")
ehr_file = st.sidebar.file_uploader("Upload EHR JSON", type=["json"])
if ehr_file is not None:
    try:
        ehr_json = json.load(ehr_file)
        st.session_state["ehr_json"] = ehr_json
        st.sidebar.success("EHR loaded ✅")
    except Exception as e:
        st.sidebar.error(f"Invalid JSON: {e}")

st.sidebar.header("My Conversations")
for thread_id in st.session_state["chat_threads"][::-1]:
    title = conversation_title(thread_id)
    if st.sidebar.button(title, key=f"btn_{thread_id}"):
        st.session_state["thread_id"] = thread_id
        msgs = load_conversation(thread_id)
        temp = []
        for msg in msgs:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            temp.append({"role": role, "content": msg.content})
        st.session_state["message_history"] = temp


# ----------------------------- Main Chat UI -----------------------------
st.title("💬 Nutrition Agent Chatbot (RAG + Tools)")

# show history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask about meal plans, diet for diabetes/CKD/HTN, affordable foods, etc...")

if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    CONFIG = {"configurable": {"thread_id": st.session_state["thread_id"]}}

    # Prepare graph input state (messages + user profile + ehr + constraints)
    graph_input = {
        "messages": [HumanMessage(content=user_input)],
        "user_profile": st.session_state["user_profile"],
        "ehr_json": st.session_state["ehr_json"],
        "ehr_ready": bool(st.session_state["ehr_json"]),
        "hard_constraints": st.session_state["hard_constraints"],
        "ehr_context": "",
        "tool_context": {},
        "final_answer": "",
    }

    with st.chat_message("assistant"):
        def ai_only_stream():
            # We stream messages emitted by the graph
            for chunk, meta in chatbot.stream(graph_input, config=CONFIG, stream_mode="messages"):
                if isinstance(chunk, AIMessage):
                    yield chunk.content

        ai_text = st.write_stream(ai_only_stream())

    st.session_state["message_history"].append({"role": "assistant", "content": ai_text})
