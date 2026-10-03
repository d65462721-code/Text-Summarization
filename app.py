import streamlit as st
from pypdf import PdfReader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Text Summarizer",
    page_icon="📝",
    layout="wide"
)


# -------------------------------
# Models
# -------------------------------
MODELS = {
    "DistilBART (faster, lighter)": "sshleifer/distilbart-cnn-12-6",
    "BART": "facebook/bart-large-cnn"
}


# -------------------------------
# Summary Lengths
# -------------------------------
LENGTHS = {
    "Short": (30, 80),
    "Medium": (60, 150),
    "Long": (100, 250)
}


CHUNK_WORDS = 600


# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource(
    show_spinner="Loading model (first run downloads it)..."
)
def load_model(model_id: str):

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    return tokenizer, model


# -------------------------------
# Split Text into Chunks
# -------------------------------
def chunk_text(text: str, size: int = CHUNK_WORDS):

    words = text.split()

    return [
        " ".join(words[i:i + size])
        for i in range(0, len(words), size)
    ]


# -------------------------------
# Summarize One Chunk
# -------------------------------
def summarize_chunk(
    text,
    tokenizer,
    model,
    min_len,
    max_len
):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    ids = model.generate(
        **inputs,
        min_length=min_len,
        max_length=max_len,
        num_beams=4,
        length_penalty=2.0,
        no_repeat_ngram_size=3,
        early_stopping=True
    )

    return tokenizer.decode(
        ids[0],
        skip_special_tokens=True
    )


# -------------------------------
# Summarize Complete Text
# -------------------------------
def summarize(
    text,
    tokenizer,
    model,
    min_len,
    max_len
):

    chunks = chunk_text(text)

    progress = st.progress(
        0.0,
        text="Summarizing..."
    )

    parts = []

    for i, chunk in enumerate(chunks, 1):

        words = len(chunk.split())

        chunk_min_len = min(
            min_len,
            max(5, words // 2)
        )

        parts.append(
            summarize_chunk(
                chunk,
                tokenizer,
                model,
                chunk_min_len,
                max_len
            )
        )

        progress.progress(
            i / len(chunks),
            text=f"Summarized chunk {i}/{len(chunks)}"
        )

    progress.empty()

    combined = " ".join(parts)

    # If there are multiple chunks, summarize the combined result
    if len(chunks) > 1 and len(combined.split()) > CHUNK_WORDS:

        combined = summarize_chunk(
            combined,
            tokenizer,
            model,
            min_len,
            max_len
        )

    return combined


# -------------------------------
# Read PDF
# -------------------------------
def read_pdf(file):

    reader = PdfReader(file)

    return "\n".join(
        page.extract_text() or ""
        for page in reader.pages
    )


# -------------------------------
# Sidebar Settings
# -------------------------------
st.sidebar.header("⚙️ Settings")


model_name = st.sidebar.selectbox(
    "Model",
    list(MODELS.keys())
)


length = st.sidebar.radio(
    "Summary length",
    list(LENGTHS.keys()),
    index=1
)


# -------------------------------
# Main Page
# -------------------------------
st.title("📝 Text Summarizer")

st.write(
    "Paste text or upload a file (.txt / .pdf) "
    "to get a concise summary."
)


# -------------------------------
# Input Type
# -------------------------------
source = st.radio(
    "Input type",
    ["Paste text", "Upload file"],
    horizontal=True
)


text = ""


# -------------------------------
# Paste Text
# -------------------------------
if source == "Paste text":

    text = st.text_area(
        "Your text",
        height=300,
        placeholder="Paste an article, notes, or report here..."
    )


# -------------------------------
# Upload File
# -------------------------------
else:

    file = st.file_uploader(
        "Upload .txt or .pdf",
        type=["txt", "pdf"]
    )

    if file:

        if file.name.lower().endswith(".pdf"):

            text = read_pdf(file)

        else:

            text = file.read().decode(
                "utf-8",
                errors="ignore"
            )

        with st.expander("📄 Preview extracted text"):

            st.write(
                text[:3000]
                + ("..." if len(text) > 3000 else "")
            )


# -------------------------------
# Summarize Button
# -------------------------------
if st.button(
    "✨ Summarize",
    type="primary"
):

    word_count = len(text.split())

    # Check minimum text
    if word_count < 50:

        st.warning(
            "Please provide at least 50 words of text."
        )

    else:

        # Load model
        with st.spinner(
            "Loading summarization model..."
        ):

            tokenizer, model = load_model(
                MODELS[model_name]
            )


        # Get summary length
        min_len, max_len = LENGTHS[length]


        # Generate summary
        summary = summarize(
            text,
            tokenizer,
            model,
            min_len,
            max_len
        )


        # -------------------------------
        # Display Original and Summary
        # -------------------------------
        col1, col2 = st.columns(2)


        # Original text
        with col1:

            st.subheader("📄 Original")

            st.caption(
                f"{word_count} words"
            )

            st.write(
                text[:5000]
                + ("..." if len(text) > 5000 else "")
            )


        # Summary
        with col2:

            st.subheader("📌 Summary")

            s_count = len(summary.split())


            if word_count > 0:

                shorter = (
                    100
                    - (s_count * 100 // word_count)
                )

            else:

                shorter = 0


            st.caption(
                f"{s_count} words · "
                f"{shorter}% shorter"
            )


            st.success(summary)


            # Download summary
            st.download_button(
                "⬇️ Download Summary",
                summary,
                file_name="summary.txt",
                mime="text/plain"
            )

