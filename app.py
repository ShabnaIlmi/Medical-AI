# Import libraries
import fitz
import streamlit as st
from groq import Groq
import os
from dotenv import load_dotenv
from pdf2image import convert_from_bytes
import pytesseract

# Tesseract path (Windows)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load environment variables
load_dotenv()

# Configure Groq API
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama3-70b-8192"

# Validate API key
if not GROQ_API_KEY or "gsk_" not in GROQ_API_KEY:
    st.error("GROQ_API_KEY is missing or invalid. Please set it in your .env file or directly in the script.")
    st.stop()

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Function to call LLaMA model on Groq
def call_llama_groq(prompt):
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a friendly AI medical assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error calling Groq: {str(e)}"

# Function to extract text from PDF (text or OCR)
def extract_text_from_pdf(uploaded_file):
    text = ""
    try:
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as pdf:
            for page in pdf:
                text += page.get_text()
    except Exception as e:
        st.error(f"PyMuPDF error: {e}")
        return ""

    if text.strip():
        return text.strip()

    st.warning("No readable text found — attempting OCR.")
    try:
        images = convert_from_bytes(uploaded_file.getvalue())
        ocr_text = ""
        for i, image in enumerate(images):
            ocr_text += f"\n--- Page {i+1} ---\n"
            ocr_text += pytesseract.image_to_string(image)
        return ocr_text.strip()
    except Exception as e:
        st.error(f"OCR failed: {e}")
        return ""

# Main UI function
def main():
    st.set_page_config(page_title="Medical Report Simplifier", layout="centered")
    st.title("Patient-Friendly Medical Report Simplifier")
    st.markdown("Upload your medical report or describe your symptoms. This tool will simplify medical language and provide practical next steps.")

    with st.expander("How this works"):
        st.info("This tool uses AI to interpret and simplify medical reports or symptom descriptions. It provides clear summaries and actionable advice in plain language.")

    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("Upload Medical Report (PDF)", type=["pdf"])
    with col2:
        user_query = st.text_area("Or describe your symptoms", height=150, placeholder="Example: I've had chest pain and fatigue for the past week...")

    st.divider()

    if st.button("Simplify and Provide Advice"):
        with st.spinner("Processing your input..."):
            input_text = ""
            source = ""

            if uploaded_file:
                input_text = extract_text_from_pdf(uploaded_file)
                source = "PDF medical report"
            elif user_query.strip():
                input_text = user_query.strip()
                source = "written description"
            else:
                st.warning("Please upload a report or enter a description.")
                return

            if not input_text:
                st.error("No readable content found.")
                return

            system_prompt = f"""
The following is a {source} from a patient:

==== START ====
{input_text}
==== END ====

Your task:
1. Simplify this information in a way a non-medical person can understand.
2. Provide useful, clear recommendations in plain language (next steps, lifestyle changes, treatments to discuss with a doctor).
Respond in a helpful and caring tone.
"""

            output = call_llama_groq(system_prompt)

        st.success("Analysis complete.")

        st.subheader("Original Input")
        st.code(input_text[:1000] + "..." if len(input_text) > 1000 else input_text)

        # Handle recommendation splitting safely
        st.subheader("Simplified Medical Summary")
        output_lower = output.lower()
        if "recommendation" in output_lower:
            index = output_lower.find("recommendation")
            simplified = output[:index].strip()
            recommendation = output[index:].strip()
        else:
            simplified = output.strip()
            recommendation = None

        st.markdown(simplified)

        if recommendation:
            st.subheader("Advice and Recommendations")
            st.markdown(recommendation)

# Run the app
if __name__ == "__main__":
    main()
