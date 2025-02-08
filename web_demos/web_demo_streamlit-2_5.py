import streamlit as st
from PIL import Image
import torch
from transformers import AutoModel, AutoTokenizer

# Model path
model_path = "openbmb/MiniCPM-Llama3-V-2_5"

# Set page configuration
st.set_page_config(
    page_title="MiniCPM-Llama3-V-2_5 Streamlit",
    page_icon=":robot:",
    layout="wide"
)

# Load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    print(f"load_model_and_tokenizer from {model_path}")
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16).to(device="cuda" if torch.cuda.is_available() else "cpu") # Handle CUDA availability
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model, st.session_state.tokenizer = load_model_and_tokenizer()
    st.session_state.model.eval()
    print("model and tokenizer loaded successfully!")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Sidebar settings
st.sidebar.title("MiniCPM-Llama3-V-2_5 Streamlit")
max_length = st.sidebar.slider("max_length", 0, 4096, 2048, step=2)
repetition_penalty = st.sidebar.slider("repetition_penalty", 0.0, 2.0, 1.05, step=0.01)
top_p = st.sidebar.slider("top_p", 0.0, 1.0, 0.8, step=0.01)
top_k = st.sidebar.slider("top_k", 0, 100, 100, step=1)
temperature = st.sidebar.slider("temperature", 0.0, 1.0, 0.7, step=0.01)

if st.sidebar.button("Clear chat history", key="clean"):
    st.session_state.chat_history = []
    st.session_state.response = ""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    st.experimental_rerun()  # Update this method to clear state

# Display chat history
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.chat_message(name="user", avatar="user").markdown(message["content"])
    else:
        st.chat_message(name="model", avatar="assistant").markdown(message["content"])

# Handle image uploads
selected_mode = st.sidebar.selectbox("Select mode", ["Text", "Image"])
if selected_mode == "Image":
    uploaded_image = st.sidebar.file_uploader("Upload image", type=["jpg", "jpeg", "png"], accept_multiple_files=False)
    if uploaded_image is not None:
        st.image(uploaded_image, caption="User uploaded image", use_column_width=True)
        st.session_state.chat_history.append({"role": "user", "content": None, "image": uploaded_image})

# User input box
user_text = st.chat_input("Enter your question")
if user_text:
    st.session_state.chat_history.append({"role": "user", "content": user_text, "image": None})
    st.chat_message(U_NAME, avatar="user").markdown(f"{U_NAME}: {user_text}")

    model = st.session_state.model
    tokenizer = st.session_state.tokenizer
    imagefile = None

    if len(st.session_state.chat_history) > 1 and st.session_state.chat_history[-2]["image"] is not None:
        imagefile = Image.open(st.session_state.chat_history[-2]["image"]).convert('RGB')

    msgs = [{"role": "user", "content": user_text}]
    res = model.chat(image=imagefile, msgs=msgs, context=None, tokenizer=tokenizer, sampling=True, top_p=top_p, top_k=top_k, repetition_penalty=repetition_penalty, temperature=temperature, stream=True)
    generated_text = st.empty().write_stream(res)  # Modify to handle stream correctly

    st.session_state.chat_history.append({"role": "model", "content": generated_text, "image": None})
    st.divider()
