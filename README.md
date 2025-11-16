# Resume Category Prediction App 📄

This project is a **Resume Category Prediction** system built using Python, Streamlit, and machine learning. It allows users to upload resumes in **PDF, DOCX, or TXT** formats and predicts the professional category of the candidate, such as **Data Science, Advocate, Network Security Engineer**, etc.  

The app uses **text preprocessing, TF-IDF vectorization, and a K-Nearest Neighbors classifier** to classify resumes into multiple categories.

---

## Features

- Upload resumes in **PDF, DOCX, or TXT** formats.
- Extract text from uploaded resumes.
- Clean resumes by removing:
  - URLs, hashtags, mentions
  - Special characters and punctuations
  - Stopwords
  - Perform lemmatization
- Predict the **category of the resume** using a trained machine learning model.
- Simple and interactive **Streamlit interface**.
- Handles multiple professional categories including technical, managerial, legal, and arts.

---

## Tech Stack

- **Python 3.10+**
- **Streamlit** – For the web interface
- **Scikit-learn** – Machine learning (KNN classifier, TF-IDF vectorizer, LabelEncoder)
- **Spacy** – NLP for stopword removal and lemmatization
- **PyPDF2** – Extract text from PDF files
- **python-docx** – Extract text from DOCX files
- **pickle** – Save and load trained models
