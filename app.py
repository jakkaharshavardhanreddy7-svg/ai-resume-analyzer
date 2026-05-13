import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Page settings
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

# Load CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Extract text from PDF
def extract_text(pdf):
    text = ""
    reader = PdfReader(pdf)

    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()

    return text

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    return text

# Title
st.title("📄 AI Resume Analyzer")

st.write("Upload your resume and compare with job description")

# Layout
col1, col2 = st.columns(2)

with col1:
    uploaded_resume = st.file_uploader(
        "Upload Resume",
        type=["pdf"]
    )

with col2:
    job_description = st.text_area(
        "Paste Job Description",
        height=250
    )

# Button
if st.button("Analyze Resume"):

    if uploaded_resume and job_description:

        # Extract resume text
        resume_text = extract_text(uploaded_resume)

        # Clean texts
        resume_text = clean_text(resume_text)
        job_description = clean_text(job_description)

        # Vectorization
        cv = CountVectorizer()

        vectors = cv.fit_transform([
            resume_text,
            job_description
        ])

        # Similarity
        similarity = cosine_similarity(vectors)[0][1]

        score = round(similarity * 100, 2)

        # Show score
        st.subheader("ATS Match Score")

        st.progress(int(score))

        st.write(f"### {score}% Match")

        # Feedback
        if score >= 80:
            st.success("Excellent Resume Match 🚀")

        elif score >= 60:
            st.warning("Good Resume — Improve Some Skills ⚡")

        else:
            st.error("Low Match — Add More Relevant Skills ❌")

        # Missing keywords
        jd_words = set(job_description.split())
        resume_words = set(resume_text.split())

        missing = jd_words - resume_words

        st.subheader("Missing Keywords")

        top_missing = list(missing)[:15]

        for word in top_missing:
            st.write("✅", word)

    else:
        st.warning("Please upload resume and enter job description.")