import streamlit as st
from PIL import Image
import torch
from transformers import AutoModel, AutoTokenizer

# Model path
model_path = "openbmb/MiniCPM-V-2"

# User and assistant names
U_NAME = "User"
A_NAME = "Assistant"

# Set page configuration
st.set_page_config(
    page_title="Minicpm-V-2 Streamlit",
    page_icon=":robot:",
    layout="wide"
)

# Load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    print(f"load_model_and_tokenizer from {model_path}")
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(
        device="cuda:0", dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model, st.session_state.tokenizer = load_model_and_tokenizer()
    print("model and tokenizer had loaded completed!")

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Sidebar settings
sidebar_name = st.sidebar.title("Minicpm-V-2 Streamlit")
max_length = st.sidebar.slider("max_length", 0, 4096, 2048, step=2)
top_p = st.sidebar.slider("top_p", 0.0, 1.0, 0.8, step=0.01)
temperature = st.sidebar.slider("temperature", 0.0, 1.0, 0.7, step=0.01)

# Clear chat history button
buttonClean = st.sidebar.button("Clear chat history", key="clean")
if buttonClean:
    st.session_state.chat_history = []
    st.session_state.response = ""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    st.rerun()

# Display chat history
for i, message in enumerate(st.session_state.chat_history):
    if message["role"] == "user":
        with st.chat_message(name="user", avatar="user"):
            if message["image"] is not None:
                st.image(message["image"], caption='User uploaded image', width=468, use_column_width=False)
                continue
            elif message["content"] is not None:
                st.markdown(message["content"])
    else:
        with st.chat_message(name="model", avatar="assistant"):
            st.markdown(message["content"])

# Select mode
selected_mode = st.sidebar.selectbox("Select mode", ["Text", "Image"])
if selected_mode == "Image":
    # Image mode
    uploaded_image = st.sidebar.file_uploader("Upload image", key=1, type=["jpg", "jpeg", "png"], accept_multiple_files=False)
    if uploaded_image is not None:
        st.image(uploaded_image, caption='User uploaded image', width=468, use_column_width=False)
        # Add uploaded image to chat history
        st.session_state.chat_history.append({"role": "user", "content": None, "image": uploaded_image})

# User input box
user_text = st.chat_input("Enter your question")
if user_text:
    with st.chat_message(U_NAME, avatar="user"):
        st.session_state.chat_history.append({"role": "user", "content": user_text, "image": None})
        st.markdown(f"{U_NAME}: {user_text}")

    # Generate reply using the model
    model = st.session_state.model
    tokenizer = st.session_state.tokenizer

    with st.chat_message(A_NAME, avatar="assistant"):
        # If the previous message contains an image, pass the image to the model
        if len(st.session_state.chat_history) > 1 and st.session_state.chat_history[-2]["image"] is not None:
            uploaded_image = st.session_state.chat_history[-2]["image"]
            imagefile = Image.open(uploaded_image).convert('RGB')

        msgs = [{"role": "user", "content": user_text}]
        res, context, _ = model.chat(image=imagefile, msgs=msgs, context=None, tokenizer=tokenizer,
                                     sampling=True,top_p=top_p,temperature=temperature)
        st.markdown(f"{A_NAME}: {res}")
        st.session_state.chat_history.append({"role": "model", "content": res, "image": None})

    st.divider()
