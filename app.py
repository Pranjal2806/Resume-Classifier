# Required installations:
# pip install streamlit scikit-learn python-docx PyPDF2

import streamlit as st
import pickle
import docx  # To handle Word files
import PyPDF2  # To handle PDF files
import re

def load_model_files():
    """Load the pre-trained model, TF-IDF vectorizer, and label encoder."""
    try:
        svc_model = pickle.load(open('clf.pkl', 'rb'))  # Ensure this file exists
        tfidf = pickle.load(open('tfidf.pkl', 'rb'))  # Ensure this file exists
        label_encoder = pickle.load(open('encoder.pkl', 'rb'))  # Ensure this file exists
        return svc_model, tfidf, label_encoder
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.stop()

# Load the model, vectorizer, and encoder
svc_model, tfidf, label_encoder = load_model_files()

def clean_resume_text(text):
    """Clean the resume text by removing unwanted characters and patterns."""
    text = re.sub(r'http\S+\s', ' ', text)
    text = re.sub(r'RT|cc', ' ', text)
    text = re.sub(r'#\S+\s', ' ', text)
    text = re.sub(r'@\S+', ' ', text)
    text = re.sub(r'[%s]' % re.escape(r"""!#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"""), ' ', text)
    text = re.sub(r'[^\x00-\x7f]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_text_from_pdf(file):
    """Extract text from a PDF file."""
    pdf_reader = PyPDF2.PdfReader(file)
    return ''.join(page.extract_text() for page in pdf_reader.pages if page.extract_text())

def extract_text_from_docx(file):
    """Extract text from a DOCX file."""
    doc = docx.Document(file)
    return '\n'.join(paragraph.text for paragraph in doc.paragraphs)

def extract_text_from_txt(file):
    """Extract text from a TXT file."""
    try:
        return file.read().decode('utf-8')
    except UnicodeDecodeError:
        return file.read().decode('latin-1')  # Fallback encoding

def handle_uploaded_file(file):
    """Handle the uploaded file and extract its text based on the file type."""
    extension = file.name.split('.')[-1].lower()
    if extension == 'pdf':
        return extract_text_from_pdf(file)
    elif extension == 'docx':
        return extract_text_from_docx(file)
    elif extension == 'txt':
        return extract_text_from_txt(file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")

def predict_resume_category(text):
    """Predict the category of the resume text."""
    cleaned_text = clean_resume_text(text)
    vectorized_text = tfidf.transform([cleaned_text])  # Vectorize the cleaned text
    dense_vectorized_text = vectorized_text.toarray()  # Convert sparse matrix to dense
    predicted_label = svc_model.predict(dense_vectorized_text)  # Predict the category
    return label_encoder.inverse_transform(predicted_label)[0]  # Decode the label

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Resume Category Prediction", page_icon="📄", layout="centered")

    st.title("Resume Category Prediction App")
    st.markdown("Upload a resume (PDF, DOCX, or TXT format), and the app will predict its category.")

    uploaded_file = st.file_uploader("Upload Your Resume", type=["pdf", "docx", "txt"])

    if uploaded_file:
        try:
            resume_text = handle_uploaded_file(uploaded_file)
            st.success("Resume text extracted successfully.")

            if st.checkbox("Show Extracted Text", value=False):
                st.text_area("Extracted Resume Text", resume_text, height=300)

            # Predict category
            st.subheader("Prediction Result")
            predicted_category = predict_resume_category(resume_text)
            st.write(f"The predicted category is: **{predicted_category}**")

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()