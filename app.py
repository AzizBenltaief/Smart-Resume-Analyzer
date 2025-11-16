import streamlit as st
import pickle 
import docx     #Extract text from Word file
import PyPDF2   #Extract text from PDF file
import re
import spacy
from pdf2image import convert_from_bytes
import pytesseract

# Load pre-trained model and TF-IDF vectorizer(they should be saved)
model = pickle.load(open('clf.pkl','rb'))
tfidf = pickle.load(open('tfidf.pkl','rb'))
le = pickle.load(open('encoder.pkl','rb'))

# Function to clean resume text
def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText


# Function to extract text from PDF
def extract_text_from_pdf(file):
    # Try normal text extraction first
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + '\n'

    # If no text extracted, fallback to OCR
    if len(text.strip()) == 0:
        # Reset file pointer to start
        file.seek(0)
        pages = convert_from_bytes(file.read())
        text = ""
        for page in pages:
            text += pytesseract.image_to_string(page) + "\n"

    return text

# Function to extract text from DOCX
def extract_text_from_word(file):
    doc = docx.Document(file)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text

# Function to extract text from TXT with explicit encoding handling
def extract_text_from_txt(file):
    try:
        text = file.read().decode('utf-8')
    except UnicodeDecodeError:
        text = file.read().decode('latin-1')
    return text

# Function to handle file upload and extraction
def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        text = extract_text_from_pdf(uploaded_file)
    elif file_extension == 'docx':
        text = extract_text_from_word(uploaded_file)
    elif file_extension == 'txt':
        text = extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX or TXT file.")
    return text



# Function to remove stopwords
nlp = spacy.load("en_core_web_sm")
def stopword_removal(text):
    doc = nlp(text)
    filtered_text = []
    for token in doc:
        if token.is_stop == False:
            filtered_text.append(token.text)
    return " ".join(filtered_text)

# Function to do Lemmatization
def lemmatization(text):
    doc = nlp(text)
    filtered_text = []
    for token in doc:
        filtered_text.append(token.lemma_.lower())
    return " ".join(filtered_text)

# Function to predict the category of a resume
def pred(input_resume):

    #Preprocess the input text
    cleaned_resume = cleanResume(input_resume)

    # Stopwords Removal
    cleaned_resume = stopword_removal(cleaned_resume)

    # Lemmatization
    cleaned_resume = lemmatization(cleaned_resume)

    # Vectorize the cleaned resume
    vectorized_text = tfidf.transform([cleaned_resume])

    # Convert sparse matrix to array
    vectorized_text = vectorized_text.toarray()

    # Predict the category
    prediction = model.predict(vectorized_text)

    # get name of the predicted category
    predicted_category_name = le.inverse_transform(prediction)

    return predicted_category_name[0]


# Streamlit app
def main():
    st.set_page_config(page_title="Resume Category Prediction",page_icon="📄",layout="wide")
    st.title("Resume Category Prediction App 📄")
    st.markdown("Upload a resume in PDF,TXT or DOCX format to predict its category.")

    #File upload section
    uploaded_file = st.file_uploader("Upload a Resume",type=["pdf","txt","docx"])
    if uploaded_file is not None:
        #Extract text from the uploaded file
        try:
            resume_text = handle_file_upload(uploaded_file)
            st.write("Successfully extracted the text from the uploaded file.")

            #Display extracted text
            if st.checkbox("Show extracted text",False):
                st.text_area("Extracted Resume Text: ",resume_text,height=300)
            
            # Make prediction
            st.subheader("Predicted Category: ")
            category = pred(resume_text)
            st.write(f"The predicted category of the uploaded resume is: **{category}**")
        
        except Exception as e:
            st.error(f"Error Processing the file: {str(e)}")

    
if __name__ == '__main__':
    main()

