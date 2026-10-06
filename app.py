import streamlit as st

from pdf_loader import extract_text_from_pdf_bytes, clean_text, chunk_text
from embeddings import EmbeddingModel
from vector_store import VectorStore
from config import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K

st.set_page_config(
    page_title="AI Resume Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Resume & Job Matching Assistant")
st.write("Analyze your resume using Hugging Face embeddings, FAISS and a local LLM.")


@st.cache_data(show_spinner=False)
def process_resume_pdf(pdf_bytes):
    if pdf_bytes is None:
        return ""
    text = clean_text(extract_text_from_pdf_bytes(pdf_bytes))
    return text


# Load models only when needed
@st.cache_resource
def load_embedding_model():
    return EmbeddingModel()


@st.cache_resource
def load_llm():
    from rag import LocalLLM
    return LocalLLM()


def fallback_match_summary(job_description, context):
    if not context.strip():
        return "No relevant resume content was found for the job description."

    context_lower = context.lower()
    job_lower = job_description.lower()
    keywords = [word.strip(".,;:-()[]{}\"") for word in job_lower.split() if len(word.strip(".,;:-()[]{}\"") ) > 3]
    matched = sorted({word for word in keywords if word and word in context_lower}, key=lambda x: len(x), reverse=True)[:8]
    missing = [word for word in keywords if word and word not in context_lower][:5]

    summary = ["### Resume-to-job match summary", ""]
    if matched:
        summary.append(f"Matching keywords: {', '.join(matched)}")
    else:
        summary.append("Matching keywords: No obvious keyword overlap detected.")
    if missing:
        summary.append(f"Potentially missing skills/keywords: {', '.join(missing)}")
    else:
        summary.append("No significant missing keywords detected from the provided resume snippet.")
    summary.append("")
    summary.append("The local LLM could not be loaded, so this is a best-effort summary based on the retrieved resume sections.")
    return "\n".join(summary)

# Session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""

# Sidebar
st.sidebar.header("Project Configuration")
st.sidebar.write("Embedding: BAAI/bge-small-en-v1.5")
st.sidebar.write("LLM: Qwen/Qwen2.5-0.5B-Instruct")
st.sidebar.info("Models load when required. Smaller model for better speed on CPU.")

# Resume upload
st.header("1. Upload Resume")
uploaded_file = st.file_uploader("Upload a PDF resume", type=["pdf"])

if st.button("Process Resume"):
    if uploaded_file is None:
        st.warning("Please upload a PDF resume first.")
    else:
        try:
            with st.spinner("Extracting resume text..."):
                text = process_resume_pdf(uploaded_file.getvalue())

            if not text:
                st.error("No text found. If this is a scanned PDF, OCR is required.")
            else:
                chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

                with st.spinner("Loading embedding model and creating vectors..."):
                    embedder = load_embedding_model()
                    embeddings = embedder.encode(chunks)

                store = VectorStore()
                store.add_documents(chunks, embeddings)

                st.session_state.vector_store = store
                st.session_state.resume_text = text

                st.success(f"Resume processed successfully! Created {len(chunks)} chunks.")

        except Exception as e:
            st.error(f"Resume processing failed: {e}")

if st.session_state.resume_text:
    with st.expander("View extracted resume"):
        st.text(st.session_state.resume_text)

# Job description
st.header("2. Job Description")
job_description = st.text_area(
    "Paste the job description",
    height=200
)

if st.button("Analyze Job Match"):
    if st.session_state.vector_store is None:
        st.warning("Process your resume first.")
    elif not job_description.strip():
        st.warning("Enter a job description.")
    else:
        try:
            with st.spinner("Retrieving relevant resume sections..."):
                embedder = load_embedding_model()
                query_vector = embedder.encode([job_description])[0]

                results = st.session_state.vector_store.search(
                    query_vector, TOP_K
                )

                context = "\n\n".join(item["text"] for item in results)

            with st.spinner("Loading local LLM and generating analysis..."):
                try:
                    llm = load_llm()
                    answer = llm.generate(
                        f"""Compare the resume with the job description.

Job description:
{job_description}

Identify matching skills, missing skills, relevant experience,
relevant projects, important keywords, and improvement suggestions.
Do not invent experience or qualifications.""",
                        context
                    )
                except Exception:
                    answer = fallback_match_summary(job_description, context)
                    st.warning("The local LLM is unavailable; showing a safe fallback analysis instead.")

            st.subheader("Job Match Analysis")
            st.markdown(answer)

        except Exception as e:
            st.error(f"Analysis failed: {e}")

# Resume Q&A
st.header("3. Ask Questions About Your Resume")
question = st.text_input("Your question")

if st.button("Ask AI"):
    if st.session_state.vector_store is None:
        st.warning("Process your resume first.")
    elif not question.strip():
        st.warning("Enter a question.")
    else:
        try:
            with st.spinner("Searching your resume..."):
                embedder = load_embedding_model()
                query_vector = embedder.encode([question])[0]

                results = st.session_state.vector_store.search(
                    query_vector, TOP_K
                )
                context = "\n\n".join(item["text"] for item in results)

            with st.spinner("Generating answer with the local LLM..."):
                try:
                    llm = load_llm()
                    answer = llm.generate(question, context)
                except Exception:
                    answer = (
                        "The local LLM is unavailable right now. "
                        "Please review the retrieved resume context below for the most relevant details.\n\n"
                        + context
                    )
                    st.warning("The local LLM is unavailable; showing the retrieved resume context instead.")

            st.subheader("Answer")
            st.write(answer)

        except Exception as e:
            st.error(f"Question answering failed: {e}")

with st.expander("Technical Architecture"):
    st.markdown("""
    PDF → PyMuPDF → Text Chunking → BGE Embeddings
    → FAISS Retrieval → Qwen LLM → Answer

    **Technologies:** Python, Streamlit, Hugging Face,
    Sentence Transformers, FAISS, Transformers and RAG.
    """)

