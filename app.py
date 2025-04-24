#!/usr/bin/env python
# coding: utf-8

# In[4]:


from fastapi import FastAPI


# In[3]:


from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re

# Load model and vectorizer
model = joblib.load("email_classifier_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = FastAPI()

class EmailInput(BaseModel):
    email: str

# PII Masking Function
def mask_pii(text):
    masked = text
    entities = []

    # Email addresses
    for match in re.finditer(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text):
        entity = match.group()
        start, end = match.span()
        masked = masked.replace(entity, "[email]")
        entities.append({
            "position": [start, end],
            "classification": "email",
            "entity": entity
        })
# Phone Number (10-digit)
    for match in re.finditer(r"\b\d{10}\b", text):
        entity = match.group()
        start, end = match.span()
        masked = masked.replace(entity, "[phone_number]")
        entities.append({
            "position": [start, end],
            "classification": "phone_number",
            "entity": entity
        })
 # Date of Birth
    for match in re.finditer(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", text):
        entity = match.group()
        start, end = match.span()
        masked = masked.replace(entity, "[dob]")
        entities.append({
            "position": [start, end],
            "classification": "dob",
            "entity": entity
        })

    # Aadhar Number (XXXX-XXXX-XXXX)
    for match in re.finditer( r"\b(?:\d{4}[-\s]?){2}\d{4}\b", text):
        entity = match.group()
        start, end = match.span()
        masked = masked.replace(entity, "[aadhar_num]")
        entities.append({
            "position": [start, end],
            "classification": "aadhar_num",
            "entity": entity
        })

    return masked, entities
      # Credit/Debit Card Number
    for match in re.finditer(r"\b(?:\d{4}[-\s]?){3}\d{4}\b", text):
        entity = match.group()
        start, end = match.span()
        masked = masked.replace(entity, "[credit_debit_no]")
        entities.append({
            "position": [start, end],
            "classification": "credit_debit_no",
            "entity": entity
        })
        # CVV (3-digit)
    for match in re.finditer(r"\b\d{3}\b", text):
        entity = match.group()
        start, end = match.span()
        # Avoid replacing phone or card numbers already done
        if entity not in [e["entity"] for e in entities]:
            masked = masked.replace(entity, "[cvv_no]")
            entities.append({
                "position": [start, end],
                "classification": "cvv_no",
                "entity": entity
            })

    # Expiry Date (MM/YY or MM/YYYY)
    for match in re.finditer(r"\b(0[1-9]|1[0-2])[\/\-](\d{2}|\d{4})\b", text):
        entity = match.group()
        start, end = match.span()
        masked = masked.replace(entity, "[expiry_no]")
        entities.append({
            "position": [start, end],
            "classification": "expiry_no",
            "entity": entity
        })

    return masked, entities


@app.post("/classify")
def classify(data: EmailInput):
    original_email = data.email
    masked_email, entities = mask_pii(original_email)

    # Vectorize the masked email
    vect = vectorizer.transform([masked_email])
    category = model.predict(vect)[0]

    return {
        "input_email_body": original_email,
        "list_of_masked_entities": entities,
        "masked_email": masked_email,
        "category_of_the_email": category
    }

from fastapi import FastAPI
from pydantic import BaseModel
import re

app = FastAPI()

class EmailInput(BaseModel):
    email: str

def mask_pii(text):
    masked = text
    entities = []

    patterns = {
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "phone_number": r"\b\d{10}\b",
        "dob": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        # Detect exactly 12 digits with optional separators (for Aadhar)
"aadhar_num":r"\b(?:\d{4}[-\s]?){2}\d{4}\b",

# Detect exactly 16 digits grouped in 4s with space/hyphen (for cards)
"credit_debit_no": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",

        "cvv_no": r"(?<!\d)\d{3}(?!\d)",
        "expiry_no": r"\b(0[1-9]|1[0-2])[\/\-](\d{2}|\d{4})\b",
        "full_name": r"\b[A-Z][a-z]+\s[A-Z][a-z]+\b"
    }

    for label, pattern in patterns.items():
        for match in re.finditer(pattern, text):
            entity = match.group()
            start, end = match.span()
            if entity in masked:
                masked = masked.replace(entity, f"[{label}]")
                entities.append({
                    "position": [start, end],
                    "classification": label,
                    "entity": entity
                })

    return masked, entities

@app.post("/classify")
def classify_email(data: EmailInput):
    email = data.email
    masked_email, masked_entities = mask_pii(email)

    lower_email = email.lower()
    if any(word in lower_email for word in ["urgent", "asap", "issue", "problem", "help"]):
        category = "Incident"
    elif any(word in lower_email for word in ["request", "need", "want", "ask"]):
        category = "Request"
    else:
        category = "General"

    return {
        "input_email_body": email,
        "list_of_masked_entities": masked_entities,
        "masked_email": masked_email,
        "category_of_the_email": category
    }

# In[ ]:




