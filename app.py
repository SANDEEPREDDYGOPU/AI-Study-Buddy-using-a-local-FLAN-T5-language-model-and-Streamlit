
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -----------------------------------------
# 1. Configure the application
# -----------------------------------------

st.set_page_config(
    page_title="AI Study Buddy",
    page_icon="📚",
    layout="centered"
)

st.title("📚 AI Study Buddy")
st.write(
    "Your personal AI tutor for Python, Machine Learning, "
    "NLP, Deep Learning, and LLMs."
)

st.info("Powered by a local pretrained FLAN-T5 model. No paid API key required.")


# -----------------------------------------
# 2. Load and cache the model
# -----------------------------------------

MODEL_NAME = "google/flan-t5-small"

@st.cache_resource(show_spinner="Loading AI model for the first time...")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.eval()

    return tokenizer, model


# -----------------------------------------
# 3. Create the question interface
# -----------------------------------------

topic = st.selectbox(
    "Choose a topic",
    [
        "Python",
        "Machine Learning",
        "Natural Language Processing",
        "Deep Learning",
        "Large Language Models"
    ]
)

question = st.text_area(
    "Enter your question",
    placeholder="Example: What is overfitting?",
    max_chars=500
)

# -----------------------------------------
# 4. Generate the answer
# -----------------------------------------

if st.button("Generate Answer", type="primary"):

    if not question.strip():
        st.warning("Please enter a question.")

    else:
        prompt = (
            "You are a helpful computer science tutor. "
            "Explain the answer simply for a beginner. "
            f"Subject: {topic}. "
            f"Question: {question.strip()} "
            "Give a clear and concise answer."
        )

        try:
            tokenizer, model = load_model()

            with st.spinner("Generating your answer..."):
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                )

                with torch.inference_mode():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=100,
                        do_sample=False
                    )

                answer = tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=True
                )

            st.subheader("💡 Answer")
            st.write(answer)

            with st.expander("View the prompt sent to the model"):
                st.write(prompt)

        except Exception as error:
            st.error(
                "The model could not generate an answer. "
                "Check the deployment logs for details."
            )
            st.caption(str(error))


# -----------------------------------------
# 5. Footer
# -----------------------------------------

st.divider()
st.caption("Mini LLM Project | Python + Transformers + Streamlit")
