import streamlit as st
import openai
from openai import OpenAI
import json
import time
from streamlit_mic_recorder import mic_recorder
import io
import os


# Initialize OpenAI client
client = OpenAI()

# Page configuration
st.set_page_config(
    page_title="Voice Topic Tree",
    page_icon="🎙️",
    layout="wide"
)

# Initialize session state
if 'tree_nodes' not in st.session_state:
    st.session_state.tree_nodes = []
    
if 'current_parent' not in st.session_state:
    st.session_state.current_parent = None
    
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
    
if 'node_counter' not in st.session_state:
    st.session_state.node_counter = 0

if 'current_topic_text' not in st.session_state:
    st.session_state.current_topic_text = ""

if 'current_topic_label' not in st.session_state:
    st.session_state.current_topic_label = None

# Functions
def transcribe_audio(audio_bytes):
    """Transcribe audio using OpenAI Whisper API"""
    try:
        # Create a BytesIO object from audio bytes
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "recording.wav"
        
        # Transcribe using Whisper
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )
        
        return transcript
    except Exception as e:
        st.error(f"Transcription error: {str(e)}")
        return None

def detect_topic_change(new_text, conversation_history, current_topic):
    """Use GPT to detect topic changes and generate topic labels"""
    try:
        # Build context from recent conversation
        recent_context = " ".join(conversation_history[-3:]) if conversation_history else ""
        
        prompt = f"""You are analyzing a conversation for topic changes.

Previous context: {recent_context}
Current topic: {current_topic if current_topic else "None (starting conversation)"}
New text: {new_text}

Determine if the new text represents a topic change from the previous context.
If this is the first topic or a clear topic shift has occurred, respond with topic_changed: true.
Provide a brief, clear topic label (2-4 words maximum) that captures the main subject.

Respond in JSON format:
{{"topic_changed": true/false, "topic_label": "brief topic name"}}"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a topic detection assistant. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        st.error(f"Topic detection error: {str(e)}")
        # Default: create new topic if this is the first one
        return {
            "topic_changed": len(conversation_history) == 0,
            "topic_label": "New Topic"
        }

def add_node(topic_label, full_text, parent_id=None):
    """Add a new node to the tree"""
    node = {
        "id": st.session_state.node_counter,
        "label": topic_label,
        "text": full_text,
        "parent_id": parent_id,
        "children": []
    }
    
    st.session_state.tree_nodes.append(node)
    st.session_state.node_counter += 1
    
    return node["id"]

def update_current_node(additional_text):
    """Append text to the current topic node"""
    if st.session_state.tree_nodes:
        current_node_id = st.session_state.current_parent
        if current_node_id is not None:
            for node in st.session_state.tree_nodes:
                if node["id"] == current_node_id:
                    node["text"] += " " + additional_text
                    break

def render_tree_node(node, level=0):
    """Recursively render tree nodes with proper indentation"""
    indent = "　" * level  # Using full-width space for indentation
    
    # Create expandable section for each node
    with st.container():
        col1, col2 = st.columns([0.8, 0.2])
        
        with col1:
            # Display node with indentation
            st.markdown(f"{indent}**📌 {node['label']}**")
        
        with col2:
            # Click button to view full text
            if st.button("View", key=f"btn_{node['id']}", use_container_width=True):
                st.session_state.selected_node = node["id"]
    
    # Render children
    children = [n for n in st.session_state.tree_nodes if n.get("parent_id") == node["id"]]
    for child in children:
        render_tree_node(child, level + 1)

def process_recording(audio_data):
    """Process recorded audio: transcribe and detect topics"""
    if audio_data is None:
        return
    
    # Extract audio bytes
    audio_bytes = audio_data['bytes']
    
    # Transcribe
    with st.spinner("Transcribing..."):
        transcript = transcribe_audio(audio_bytes)
    
    if transcript and len(transcript.strip()) > 0:
        st.success(f"Transcribed: {transcript[:100]}...")
        
        # Detect topic change
        with st.spinner("Analyzing topic..."):
            topic_info = detect_topic_change(
                transcript,
                st.session_state.conversation_history,
                st.session_state.current_topic_label
            )
        
        # Update tree structure
        if topic_info['topic_changed']:
            # Create new topic node
            node_id = add_node(
                topic_info['topic_label'],
                transcript,
                st.session_state.current_parent
            )
            st.session_state.current_parent = node_id
            st.session_state.current_topic_label = topic_info['topic_label']
            st.info(f"📝 New topic detected: **{topic_info['topic_label']}**")
        else:
            # Append to current topic
            update_current_node(transcript)
            st.info(f"➕ Added to current topic: **{st.session_state.current_topic_label}**")
        
        # Add to conversation history
        st.session_state.conversation_history.append(transcript)

# Main UI
st.title("🎙️ Real-Time Voice Topic Tree")
st.markdown("Speak into your microphone, and the app will organize your speech into a topic tree!")

# Two-column layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🎤 Record Audio")
    
    # Audio recorder component
    audio = mic_recorder(
        start_prompt="🔴 Start Recording",
        stop_prompt="⏹️ Stop Recording",
        just_once=True,
        use_container_width=True,
        key='recorder'
    )
    
    # Process audio when recorded
    if audio is not None:
        # Check if this is new audio
        if 'last_audio_id' not in st.session_state or audio['id'] != st.session_state.last_audio_id:
            st.session_state.last_audio_id = audio['id']
            process_recording(audio)
            st.rerun()
    
    # Show conversation history
    st.subheader("📝 Conversation History")
    if st.session_state.conversation_history:
        for i, text in enumerate(st.session_state.conversation_history[-5:], 1):
            st.text_area(
                f"Recording {len(st.session_state.conversation_history) - 5 + i}",
                value=text,
                height=80,
                disabled=True,
                key=f"history_{i}"
            )
    else:
        st.info("No recordings yet. Start speaking!")
    
    # Reset button
    if st.button("🔄 Reset All", use_container_width=True):
        st.session_state.tree_nodes = []
        st.session_state.current_parent = None
        st.session_state.conversation_history = []
        st.session_state.node_counter = 0
        st.session_state.current_topic_label = None
        if 'selected_node' in st.session_state:
            del st.session_state.selected_node
        st.rerun()

with col2:
    st.subheader("🌲 Topic Tree")
    
    # Display tree structure
    if st.session_state.tree_nodes:
        # Find root nodes (nodes without parents)
        root_nodes = [n for n in st.session_state.tree_nodes if n.get("parent_id") is None]
        
        # Render tree
        for root in root_nodes:
            render_tree_node(root)
        
        st.markdown("---")
        
        # Display selected node's full text
        st.subheader("💬 Full Transcript")
        if 'selected_node' in st.session_state:
            selected = next(
                (n for n in st.session_state.tree_nodes if n["id"] == st.session_state.selected_node),
                None
            )
            if selected:
                st.markdown(f"**Topic:** {selected['label']}")
                st.text_area(
                    "Full text:",
                    value=selected['text'],
                    height=250,
                    disabled=True,
                    key="full_text_display"
                )
        else:
            st.info("Click 'View' on any topic to see the full transcript")
    else:
        st.info("🎙️ Start recording to build your topic tree!")
        st.markdown("""
        **How it works:**
        1. Click 'Start Recording' and speak
        2. Click 'Stop Recording' when done
        3. The app will transcribe your speech
        4. Topics are automatically detected and organized
        5. Click 'View' on any topic to see the full text
        """)

# Sidebar with instructions
with st.sidebar:
    st.header("ℹ️ Instructions")
    st.markdown("""
    ### How to use:
    1. **Record**: Click the red button to start recording
    2. **Speak**: Talk about any topic
    3. **Stop**: Click again to stop recording
    4. **Automatic**: The app will:
       - Transcribe your speech
       - Detect topic changes
       - Organize into a tree structure
    5. **View**: Click 'View' to see full transcripts
    
    ### Tips:
    - Speak clearly for better transcription
    - Pause between different topics
    - The app detects topic shifts automatically
    - You can record multiple times
    """)
    
    st.markdown("---")
    st.markdown("**Stats:**")
    st.metric("Total Topics", len(st.session_state.tree_nodes))
    st.metric("Total Recordings", len(st.session_state.conversation_history))

