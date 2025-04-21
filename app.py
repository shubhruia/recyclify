import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from datetime import datetime
import requests
import csv
from io import BytesIO
from groq import ask_groq

st.set_page_config(page_title="Recyclify", page_icon="♻️", layout="centered")

# Constants
MODEL_PATH = "waste_classifier_weights.weights.h5"
CLASS_LABELS = ['glass', 'metal', 'organic', 'paper', 'plastic', 'trash']
CATEGORY_ICONS = {'glass': '🍷', 'metal': '🥫', 'organic': '🌿',
                  'paper': '📄', 'plastic': '🧴', 'trash': '🗑️'}
RECYCLABILITY_SCORES = {'glass': "♻️ High", 'metal': "♻️ High", 'organic': "♻️ Medium",
                        'paper': "♻️ Medium", 'plastic': "♻️ Low", 'trash': "♻️ Very Low"}

# Load model with optimizations
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# Initialize session state
def init_session_state():
    for key in ["predicted", "image", "input_choice", "result"]:
        if key not in st.session_state:
            st.session_state[key] = False if key == "predicted" else None

init_session_state()

# Reset
def reset_app():
    for key in ["predicted", "image", "input_choice", "result"]:
        st.session_state[key] = False if key == "predicted" else None

# Prediction function
def predict_image(img_input, preprocessed=False, return_all=False):
    if not preprocessed:
        from tensorflow.keras.preprocessing import image
        img = image.load_img(img_input, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
    else:
        img_array = img_input

    predictions = model.predict(img_array)
    predicted_class = CLASS_LABELS[np.argmax(predictions)]
    confidence = round(100 * np.max(predictions), 2)

    return (predicted_class, confidence, predictions) if return_all else (predicted_class, confidence)

# Streamlit UI
st.markdown("""
# ♻️ Recyclify: A Smart Waste Classifier
Get instant waste classification, sustainability tips, and recycling advice to make a positive environmental impact. 🌱
""")

# Tabs
tab1, tab2 = st.tabs(["🗑️ Waste Classifier", "💬 Chatbot Assistant"])

with tab1:
    # Image Input Method
    if not st.session_state.predicted:
        if st.session_state.input_choice is None:
            st.session_state.input_choice = "Upload from device"

        st.subheader("📸 How do you want to provide an image?")
        input_choice = st.radio(
            "Choose image input method:",
            ["Upload from device", "Take a photo with camera", "Provide image URL"],
            index=["Upload from device", "Take a photo with camera", "Provide image URL"].index(st.session_state.input_choice),
            horizontal=True,
            key="input_method_radio"
        )

        if input_choice != st.session_state.input_choice:
            st.session_state.input_choice = input_choice
            st.rerun()

        if input_choice == "Upload from device":
            uploaded_file = st.file_uploader("📤 Upload image", type=["jpg", "jpeg", "png"])
            if uploaded_file:
                st.session_state.image = Image.open(uploaded_file).convert("RGB")

        elif input_choice == "Take a photo with camera":
            camera_image = st.camera_input("📸 Capture image")
            if camera_image:
                st.session_state.image = Image.open(camera_image).convert("RGB")

        elif input_choice == "Provide image URL":
            image_url = st.text_input("🔗 Enter image URL")
            if image_url:
                try:
                    response = requests.get(image_url)
                    st.session_state.image = Image.open(BytesIO(response.content)).convert("RGB")
                except Exception as e:
                    st.error(f"Failed to load image from URL: {e}")

    # Classification
    if st.session_state.image and not st.session_state.predicted:
        st.image(st.session_state.image, use_container_width=True)

        if st.button("Classify this image"):
            with st.spinner("Analyzing the image..."):
                img_array = np.array(st.session_state.image.resize((224, 224))) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                label, confidence, all_probs = predict_image(img_array, preprocessed=True, return_all=True)

            st.session_state.predicted = True
            st.session_state.result = (label, confidence, all_probs)

    # Result
    if st.session_state.predicted:
        label, confidence, all_probs = st.session_state.result
        icon = CATEGORY_ICONS[label]
        st.success(f"This image contains **{label}** {icon} ({confidence:.2f}%)")

        # Confidence Chart
        st.subheader("Confidence for Each Category")
        prob_percentages = {
            f"{CATEGORY_ICONS[cat]} {cat}": round(prob * 100, 2)
            for cat, prob in zip(CLASS_LABELS, all_probs[0])
        }
        st.bar_chart(prob_percentages)

        # Recyclability
        st.markdown("### Recyclability Score")
        st.info(RECYCLABILITY_SCORES[label])

        # Environmental Fact
        fact_prompt = f"Give an short interesting environmental fact about the {label} waste category. Keep it short but insightful."
        st.markdown("### 💡Fun Fact!")
        st.warning(ask_groq(fact_prompt))

        # Eco Tip
        tip_prompt = f"Suggest a concise general eco-friendly waste management tip related to {label} waste. Focus on sustainable actions."
        st.markdown("### 🌱 Eco-Friendly Suggestion")
        st.info(ask_groq(tip_prompt))

        # Feedback
        st.markdown("### 📝 Feedback")
        feedback = st.selectbox("Was this prediction correct?", ["Select an option", "Yes", "No"])

        if feedback == "Yes":
            st.success("🎯 Awesome! The model got it right. Thanks for confirming!")
        elif feedback == "No":
            correct_label = st.selectbox("Select the correct category:", CLASS_LABELS)
            if st.button("✅ Submit Feedback"):
                timestamp = datetime.now().isoformat()
                input_method = st.session_state.input_choice.lower().split()[0]
                data = [timestamp, label, f"{confidence:.2f}", correct_label, input_method]
                header = ["timestamp", "predicted", "confidence", "corrected", "method"]

                try:
                    with open("feedback_log.csv", "a", newline="") as f:
                        writer = csv.writer(f)
                        if f.tell() == 0:
                            writer.writerow(header)
                            writer.writerow(data)
                        st.success("✅ Feedback submitted. Thank you!")
                except Exception as e:
                    st.error(f"Error saving feedback: {e}")

        # Restart
        st.markdown("### ♻️ Classify something else?")
        if st.button("Try Another"):
            reset_app()
            st.rerun()

def chatbot_tab():
    SYSTEM_INSTRUCTION = (
        "You are an eco assistant tailored for Indian users. "
        "Provide India-specific advice, data, and examples. "
        "Base your suggestions on Indian culture, laws, and environmental practices. "
        "Avoid global generalizations unless asked explicitly. Keep the tone simple, clear, and friendly."
    )

    st.header("💬 Eco Assistant")
    st.markdown("Ask me anything about recycling, waste categories, or how to be more eco-friendly.")

    # Initialize chat history and flags
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{
            "role": "assistant",
            "content": "👋 Hi! I'm your eco assistant. Ask me anything about waste management, recycling, or sustainable living."
        }]
    if "awaiting_response" not in st.session_state:
        st.session_state.awaiting_response = False

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Handle user input
    if not st.session_state.awaiting_response:
        user_input = st.chat_input("Type your question here...")
        if user_input and user_input.strip():
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.session_state.awaiting_response = True
            st.rerun()

    # Handle assistant response
    elif st.session_state.chat_history[-1]["role"] == "user":
        user_input = st.session_state.chat_history[-1]["content"]
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                prompt = f"{SYSTEM_INSTRUCTION}\nUser: {user_input}"
                response = ask_groq(prompt)
                if isinstance(response, tuple):
                    response = response[0]
                st.markdown(response)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.session_state.awaiting_response = False
        st.rerun()

    # Reset chat option
    with st.expander("🧹 Clear Chat Conversation"):
        if st.button("Reset Conversation"):
            st.session_state.chat_history = [{
                "role": "assistant",
                "content": "👋 Hi! I'm your eco assistant. Ask me anything about waste management, recycling, or sustainable living."
            }]
            st.session_state.awaiting_response = False
            st.rerun()

with tab2:
    chatbot_tab()

# Sidebar
with st.sidebar:
    st.markdown("## About This App")
    st.markdown("""
Helps users identify the type of waste in an image and provides helpful disposal tips to encourage proper recycling habits.
- Built using **MobileNetV2**, a lightweight CNN model ideal for its efficiency and high performance.
- Achieved an accuracy of **91.87%** on validation data.
- Trained on the [Recyclable and Household Waste Classification dataset](https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification/data), containing **15,000+ images**.
- Uses LLM to generate fun environmental facts and eco-friendly tips based on the predicted waste type.
    """)
