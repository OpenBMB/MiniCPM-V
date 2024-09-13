import os.path

import streamlit as st
import torch
from PIL import Image
from decord import VideoReader, cpu
import numpy as np
from transformers import AutoModel, AutoTokenizer

# Model path
model_path = "openbmb/MiniCPM-V-2_6"
upload_path = ".\\uploads"

# User and assistant names
U_NAME = "User"
A_NAME = "Assistant"

# Set page configuration
st.set_page_config(
    page_title="MiniCPM-V-2_6 Streamlit",
    page_icon=":robot:",
    layout="wide"
)


# Load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    print(f"load_model_and_tokenizer from {model_path}")
    model = (AutoModel.from_pretrained(model_path, trust_remote_code=True, attn_implementation='sdpa').
             to(dtype=torch.bfloat16))
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer


# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model, st.session_state.tokenizer = load_model_and_tokenizer()
    st.session_state.model.eval().cuda()
    print("model and tokenizer had loaded completed!")

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.uploaded_image_list = []
    st.session_state.uploaded_image_num = 0
    st.session_state.uploaded_video_list = []
    st.session_state.uploaded_video_num = 0
    st.session_state.response = ""

# Sidebar settings
sidebar_name = st.sidebar.title("MiniCPM-V-2_6 Streamlit")
max_length = st.sidebar.slider("max_length", 0, 4096, 2048, step=2)
repetition_penalty = st.sidebar.slider("repetition_penalty", 0.0, 2.0, 1.05, step=0.01)
top_k = st.sidebar.slider("top_k", 0, 100, 100, step=1)
top_p = st.sidebar.slider("top_p", 0.0, 1.0, 0.8, step=0.01)
temperature = st.sidebar.slider("temperature", 0.0, 1.0, 0.7, step=0.01)

# Button to clear session history
buttonClean = st.sidebar.button("Clearing session history", key="clean")
if buttonClean:
    # Reset the session state history and uploaded file lists
    st.session_state.chat_history = []
    st.session_state.uploaded_image_list = []
    st.session_state.uploaded_image_num = 0
    st.session_state.uploaded_video_list = []
    st.session_state.uploaded_video_num = 0
    st.session_state.response = ""

    # If using GPU, clear the CUDA cache to free up memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Rerun to refresh the interface
    st.rerun()

# Display chat history
for i, message in enumerate(st.session_state.chat_history):
    if message["role"] == "user":
        with st.chat_message(name="user", avatar="user"):
            if message["image"] is not None:
                st.image(message["image"], caption='User uploaded images', width=512, use_column_width=False)
                continue
            elif message["video"] is not None:
                st.video(message["video"], format="video/mp4", loop=False, autoplay=False, muted=True)
                continue
            elif message["content"] is not None:
                st.markdown(message["content"])
    else:
        with st.chat_message(name="model", avatar="assistant"):
            st.markdown(message["content"])

# Select mode
selected_mode = st.sidebar.selectbox("Select Mode", ["Text", "Single Image", "Multiple Images", "Video"])

# Supported image file extensions
image_type = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']

if selected_mode == "Single Image":
    # Single Image Mode
    uploaded_image = st.sidebar.file_uploader("Upload a Single Image", key=1, type=image_type,
                                              accept_multiple_files=False)
    if uploaded_image is not None:
        st.image(uploaded_image, caption='User Uploaded Image', width=512, use_column_width=False)
        # Add the uploaded image to the chat history
        st.session_state.chat_history.append({"role": "user", "content": None, "image": uploaded_image, "video": None})
        st.session_state.uploaded_image_list = [uploaded_image]
        st.session_state.uploaded_image_num = 1

if selected_mode == "Multiple Images":
    # Multiple Images Mode
    uploaded_image_list = st.sidebar.file_uploader("Upload Multiple Images", key=2, type=image_type,
                                                   accept_multiple_files=True)
    uploaded_image_num = len(uploaded_image_list)

    if uploaded_image_list is not None and uploaded_image_num > 0:
        for img in uploaded_image_list:
            st.image(img, caption='User Uploaded Image', width=512, use_column_width=False)
            # Add the uploaded images to the chat history
            st.session_state.chat_history.append({"role": "user", "content": None, "image": img, "video": None})
        # Update the uploaded image list and count in st.session_state
        st.session_state.uploaded_image_list = uploaded_image_list
        st.session_state.uploaded_image_num = uploaded_image_num

# Supported video format suffixes
video_type = ['.mp4', '.mkv', '.mov', '.avi', '.flv', '.wmv', '.webm', '.m4v']

# Tip: You can use the command `streamlit run ./web_demo_streamlit-minicpmv2_6.py --server.maxUploadSize 1024`
# to adjust the maximum upload size to 1024MB or larger files.
# The default 200MB limit of Streamlit's file_uploader component might be insufficient for video-based interactions.
# Adjust the size based on your GPU memory usage.

if selected_mode == "Video":
    # 单个视频模态
    uploaded_video = st.sidebar.file_uploader("Upload a single video file", key=3, type=video_type,
                                              accept_multiple_files=False)
    if uploaded_video is not None:
        st.video(uploaded_video, format="video/mp4", loop=False, autoplay=False, muted=True)
        st.session_state.chat_history.append({"role": "user", "content": None, "image": None, "video": uploaded_video})

        uploaded_video_path = os.path.join(upload_path, uploaded_video.name)
        with open(uploaded_video_path, "wb") as vf:
            vf.write(uploaded_video.getvalue())
        st.session_state.uploaded_video_list = [uploaded_video_path]
        st.session_state.uploaded_video_num = 1

MAX_NUM_FRAMES = 64  # if cuda OOM set a smaller number


# Encodes a video by sampling frames at a fixed rate and converting them to image arrays.
def encode_video(video_path):
    def uniform_sample(frame_indices, num_samples):
        # Calculate sampling interval and uniformly sample frame indices
        gap = len(frame_indices) / num_samples
        sampled_idxs = np.linspace(gap / 2, len(frame_indices) - gap / 2, num_samples, dtype=int)
        return [frame_indices[i] for i in sampled_idxs]

    # Read the video and set the decoder's context to CPU
    vr = VideoReader(video_path, ctx=cpu(0))

    # Calculate the sampling interval to sample video frames at 1 FPS
    sample_fps = round(vr.get_avg_fps() / 1)  # Use integer FPS
    frame_idx = list(range(0, len(vr), sample_fps))

    # If the number of sampled frames exceeds the maximum limit, uniformly sample them
    if len(frame_idx) > MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)

    # Retrieve the sampled frames and convert them to image arrays
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(frame.astype('uint8')) for frame in frames]

    print('Number of frames:', len(frames))
    return frames



# User input box
user_text = st.chat_input("Enter your question")
if user_text is not None:
    if user_text.strip() is "":
        st.warning('Input message could not be empty!', icon="⚠️")
    else:
        # Display user input and save it to session history
        with st.chat_message(U_NAME, avatar="user"):
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_text,
                "image": None,
                "video": None
            })
            st.markdown(f"{U_NAME}: {user_text}")

        # Generate responses using the model
        model = st.session_state.model
        tokenizer = st.session_state.tokenizer
        content_list = []  # Store the content (text or image) that will be passed into the model
        imageFile = None

        with st.chat_message(A_NAME, avatar="assistant"):
            # Handle different inputs depending on the mode selected by the user
            if selected_mode == "Single Image":
                # Single image mode: pass in the last uploaded image
                print("Single Images mode in use")
                if len(st.session_state.chat_history) > 1 and len(st.session_state.uploaded_image_list) >= 1:
                    uploaded_image = st.session_state.uploaded_image_list[-1]
                    if uploaded_image:
                        imageFile = Image.open(uploaded_image).convert('RGB')
                        content_list.append(imageFile)
                else:
                    print("Single Images mode: No image found")

            elif selected_mode == "Multiple Images":
                # Multi-image mode: pass in all the images uploaded last time
                print("Multiple Images mode in use")
                if len(st.session_state.chat_history) > 1 and st.session_state.uploaded_image_num >= 1:
                    for uploaded_image in st.session_state.uploaded_image_list:
                        imageFile = Image.open(uploaded_image).convert('RGB')
                        content_list.append(imageFile)
                else:
                    print("Multiple Images mode: No image found")

            elif selected_mode == "Video":
                # Video mode: pass in slice frames of uploaded video
                print("Video mode in use")
                if len(st.session_state.chat_history) > 1 and st.session_state.uploaded_video_num == 1:
                    uploaded_video_path = st.session_state.uploaded_video_list[-1]
                    if uploaded_video_path:
                        with st.spinner('Encoding your video, please wait...'):
                            frames = encode_video(uploaded_video_path)
                else:
                    print("Video Mode: No video found")

            # Defining model parameters
            params = {
                'sampling': True,
                'top_p': top_p,
                'top_k': top_k,
                'temperature': temperature,
                'repetition_penalty': repetition_penalty,
                "max_new_tokens": max_length,
                "stream": True
            }

            # Set different input parameters depending on whether to upload a video
            if st.session_state.uploaded_video_num == 1 and selected_mode == "Video":
                msgs = [{"role": "user", "content": frames + [user_text]}]
                # Set decode params for video
                params["max_inp_length"] = 4352  # Set the maximum input length of the video mode
                params["use_image_id"] = False  # Do not use image_id
                params["max_slice_nums"] = 1  # # use 1 if cuda OOM and video resolution >  448*448
            else:
                content_list.append(user_text)
                msgs = [{"role": "user", "content": content_list}]

            print("content_list:", content_list)  # debug
            print("params:", params)  # debug

            # Generate and display the model's responses
            with st.spinner('AI is thinking...'):
                response = model.chat(image=None, msgs=msgs, context=None, tokenizer=tokenizer, **params)
            st.session_state.response = st.write_stream(response)
            st.session_state.chat_history.append({
                "role": "model",
                "content": st.session_state.response,
                "image": None,
                "video": None
            })

        st.divider()  # Add separators to the interface

