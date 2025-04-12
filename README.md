
# Resume-classification-tool


This project helps HR professionals and recruiters automatically classify resumes into relevant job categories using a machine learning model trained on real-world data. Upload any resume in .pdf, .docx, or .txt format, and the app will predict the most suitable job role from 25 categories.


## Project overview

### 🔧  Working of the Project

Step-by-Step Flow:

File Upload: User uploads resumes in .pdf, .docx, or .txt format.

1. Text Extraction: Text is extracted based on file type.

2. Text Cleaning: The text is cleaned by removing links, special characters, and redundant spaces.

3. Vectorization: Cleaned text is converted into numerical data using TF-IDF.

4. Prediction: KNN model classifies the resume into one of 25 predefined job categories.

5. Output: The predicted job role is displayed and exportable as CSV.

###  Implementation Details
🔍 Preprocessing:
Removal of stop words, URLs, and special characters.

Normalization of text using regex and whitespace stripping.

#### 🧰 Machine Learning: 

train KNN Classifier

TF-IDF Vectorizer for text features

Label Encoder to decode predicted classes

#### 📁 File Handling:
PyPDF2 for PDFs

python-docx for DOCX

Standard Python file handling for TXT files

#### 📊 Output:
Real-time classification display on the interface

Export to CSV for HR use

## Installation

Install my-project with npm

```bash
 pip install streamlit scikit-learn PyPDF2 python-docx
 python -m streamlit run newapp.py
```
    
