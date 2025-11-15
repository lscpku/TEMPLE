import streamlit as st
import json
import glob
import os
from pathlib import Path
import pandas as pd
import re


def postprocess_caption(caption, model):
    if model.startswith("mimovl"):
        start_idxs = [caption.lower().rfind(s) for s in ["the video opens", "the video starts", "the video begins"]]
        start_idx = max(start_idxs)
        if start_idx != -1: 
            caption = caption[start_idx:]
        else:
            caption = "N/A"

        # remove all parentheses containing the word "aption" but not containing ( and )
        caption = re.sub(r'\([^()]*aption[^()]*\)', '', caption)
        caption = caption.strip()
    return caption


# Set page config
st.set_page_config(
    page_title="DPO Data Viewer",
    page_icon="📊",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
    .stSelectbox, .stMultiSelect {
        min-width: 200px;
    }
    .caption-box {
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
        font-family: monospace;
    }
    .video-container {
        width: 100%;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Constants
MODELS = ["qwen25vl7b", "internvl3-8b", "mimovl"]
CORRUPTION_TYPES = ["none", "shuffle", "reverse", "drop"]
CORRUPTION_STRENGTHS = [2, 4, 8, 16]
DATA_ROOT = "./data/lv178k_2_3m_ytb"
VIDEO_METADATA_PATH = os.path.join(DATA_ROOT, "anno/videos.jsonl")

@st.cache_data
def load_video_metadata():
    """Load and cache video metadata."""
    video_data = {}
    try:
        with open(VIDEO_METADATA_PATH, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                video_data[data['video_id']] = data
        return video_data
    except Exception as e:
        st.error(f"Error loading video metadata: {str(e)}")
        return {}

def get_data_path(model, corruption_type, strength):
    """Generate the path pattern for data files."""
    if corruption_type == "none":
        pattern = f"{DATA_ROOT}/dpo-{model}/iter_0/caption_caption_none_1_False_False.jsonl.*"
    else:
        pattern = f"{DATA_ROOT}/dpo-{model}/iter_0/caption_caption_{corruption_type}_{strength}_False_False.jsonl.*"
    return pattern

def load_data(file_pattern, num_samples=50):
    """Load data from JSONL files."""
    data = []
    try:
        files = glob.glob(file_pattern)
        if not files:
            st.error(f"No files found matching pattern: {file_pattern}")
            return []
        
        for file in files:
            with open(file, 'r') as f:
                for i, line in enumerate(f):
                    if i >= num_samples:
                        break
                    try:
                        item = json.loads(line.strip())
                        # Extract video_id from the data if available
                        if isinstance(item, dict) and 'video_id' in item:
                            data.append(item)
                        else:
                            data.append(item)
                    except json.JSONDecodeError:
                        continue
            if len(data) >= num_samples:
                break
                
        return data[:num_samples]
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return []

# Load video metadata
video_metadata = load_video_metadata()

# Sidebar controls
with st.sidebar:
    st.title("DPO Data Viewer")
    
    model = st.selectbox(
        "Select Model",
        options=MODELS,
        key="model"
    )
    
    corruption_type = st.selectbox(
        "Corruption Type",
        options=CORRUPTION_TYPES,
        key="corruption_type"
    )
    
    strength = st.selectbox(
        "Corruption Strength",
        options=CORRUPTION_STRENGTHS,
        key="strength",
        disabled=(corruption_type == "none")
    )
    
    num_samples = st.slider(
        "Number of Samples",
        min_value=1,
        max_value=100,
        value=20,
        key="num_samples"
    )

# Main content
st.header("Data Visualization")

# Load and display data
positive_pattern = get_data_path(model, "none", 1)
negative_pattern = get_data_path(model, corruption_type, strength)

with st.spinner("Loading data..."):
    positive_data = load_data(positive_pattern, num_samples)
    negative_data = load_data(negative_pattern, num_samples)

if not positive_data or not negative_data:
    st.warning("No data available for the selected configuration.")
else:
    # Display data in a three-column layout
    for i, (pos, neg) in enumerate(zip(positive_data, negative_data)):
        st.subheader(f"Sample {i+1}")
        
        # Get video information if available
        video_id = pos.get('video_id')
        if video_id and video_id in video_metadata:
            video_info = video_metadata[video_id]
            video_path = video_info['video_path']
            
            # Display video if available
            if os.path.exists(video_path):
                st.video(video_path)
            else:
                st.warning(f"Video file not found: {video_path}")
        
        # Display captions
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Positive Caption:**")
            st.markdown(f'<div class="caption-box">{postprocess_caption(pos.get("response", "N/A"), model)}</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown("**Negative Caption:**")
            st.markdown(f'<div class="caption-box">{postprocess_caption(neg.get("response", "N/A"), model)}</div>', unsafe_allow_html=True)
        
        st.divider()

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ using Streamlit") 