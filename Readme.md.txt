# Email Classification & PII Masking API

This project detects and masks sensitive information (PII/PCI) from email content and classifies the email into a support category using a machine learning model. The API is built using FastAPI.

---

## 🔐 Features
- Masks the following PII/PCI fields:
  - Full Name
  - Email Address
  - Phone Number
  - Date of Birth
  - Aadhar Card Number
  - CVV
  - Expiry Date
- Classifies masked emails into categories like:
  - Incident
  - Request
  - General

---

## 🛠 How to Run the App Locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
2. Start the server:
uvicorn app:app --reload

3. Open Swagger UI to test:
http://127.0.0.1:8000/docs

🚀 API Example
Input:

json
{
  "email": "Hi, my name is John Doe. Email: john@example.com. Phone: 9876543210.CVV: 123. Expiry: 12/26"
}

OUTPUT

{
  "input_email_body": "...",
  "masked_email": "...",
  "list_of_masked_entities": [...],
  "category_of_the_email": "Request"
}


📦 Project Files
app.py: FastAPI app and API route

utils.py: Masking logic

models.py: Loads model and vectorizer

model.pkl: Trained ML model

vectorizer.pkl: TF-IDF vectorizer

requirements.txt: Libraries used
