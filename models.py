{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "4c45329f-37eb-44f6-8c8b-fc5bf62171f2",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import classification_report\n",
    "import joblib\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "3f4bb8b8-45d1-4c92-9dcb-32d63cc51d3f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>email</th>\n",
       "      <th>type</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Subject: Unvorhergesehener Absturz der Datenan...</td>\n",
       "      <td>Incident</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Subject: Customer Support Inquiry\\n\\nSeeking i...</td>\n",
       "      <td>Request</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Subject: Data Analytics for Investment\\n\\nI am...</td>\n",
       "      <td>Request</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Subject: Krankenhaus-Dienstleistung-Problem\\n\\...</td>\n",
       "      <td>Incident</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Subject: Security\\n\\nDear Customer Support, I ...</td>\n",
       "      <td>Request</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                               email      type\n",
       "0  Subject: Unvorhergesehener Absturz der Datenan...  Incident\n",
       "1  Subject: Customer Support Inquiry\\n\\nSeeking i...   Request\n",
       "2  Subject: Data Analytics for Investment\\n\\nI am...   Request\n",
       "3  Subject: Krankenhaus-Dienstleistung-Problem\\n\\...  Incident\n",
       "4  Subject: Security\\n\\nDear Customer Support, I ...   Request"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import pandas as pd\n",
    "\n",
    "# Load your dataset (update path if needed)\n",
    "df = pd.read_csv(\"combined_emails_with_natural_pii.csv\")\n",
    "\n",
    "# Check it's loaded\n",
    "df.head()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "73a45796-55d4-44ba-94d8-a10dd3877f67",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Reuse your dataset\n",
    "def mask_email(email):\n",
    "    \"\"\"\n",
    "    Mask the email by hiding part of the username and domain.\n",
    "    Returns the masked email and the original (for logging if needed).\n",
    "    \"\"\"\n",
    "    try:\n",
    "        username, domain = email.split('@')\n",
    "        masked_username = username[0] + '*' * (len(username) - 2) + username[-1] if len(username) > 2 else '*' * len(username)\n",
    "        domain_parts = domain.split('.')\n",
    "        masked_domain = domain_parts[0][0] + '*' * (len(domain_parts[0]) - 1)\n",
    "        masked_domain += '.' + '.'.join(domain_parts[1:])\n",
    "        masked_email = f\"{masked_username}@{masked_domain}\"\n",
    "        return masked_email, email\n",
    "    except Exception as e:\n",
    "        return \"invalid_email\", email\n",
    "\n",
    "masked_emails = []\n",
    "for email in df['email']:\n",
    "    masked, _ = mask_email(email)\n",
    "    masked_emails.append(masked)\n",
    "\n",
    "# Add masked version as a new column\n",
    "df['masked_email'] = masked_emails\n",
    "\n",
    "# Define features and labels\n",
    "X = df['masked_email']\n",
    "y = df['type']  # this is your category/label column\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "1f329834-dedd-496c-9028-c5ca9e21709e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Index(['email', 'type', 'masked_email'], dtype='object')\n"
     ]
    }
   ],
   "source": [
    "print(df.columns)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "4db9f954-6b41-4152-b3ab-fc83a50c453e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "      Change       0.88      0.59      0.71       479\n",
      "    Incident       0.65      0.92      0.76      1920\n",
      "     Problem       0.45      0.14      0.21      1009\n",
      "     Request       0.87      0.90      0.89      1392\n",
      "\n",
      "    accuracy                           0.72      4800\n",
      "   macro avg       0.71      0.64      0.64      4800\n",
      "weighted avg       0.70      0.72      0.68      4800\n",
      "\n"
     ]
    }
   ],
   "source": [
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.naive_bayes import MultinomialNB\n",
    "from sklearn.metrics import classification_report\n",
    "\n",
    "# Use the actual email body\n",
    "X = df['email']  # this is the column with email content\n",
    "y = df['type']   # assuming this column contains categories like 'Billing Issues', etc.\n",
    "\n",
    "# Vectorize the text\n",
    "vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)\n",
    "X_vect = vectorizer.fit_transform(X)\n",
    "\n",
    "# Split data\n",
    "X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)\n",
    "\n",
    "# Train the model\n",
    "model = MultinomialNB()\n",
    "model.fit(X_train, y_train)\n",
    "\n",
    "# Evaluate\n",
    "y_pred = model.predict(X_test)\n",
    "print(classification_report(y_test, y_pred))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "96eab9fe-25e8-419e-981c-0595a4074952",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['vectorizer.pkl']"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import joblib\n",
    "\n",
    "# Save the trained model and vectorizer\n",
    "joblib.dump(model, \"email_classifier_model.pkl\")\n",
    "joblib.dump(vectorizer, \"vectorizer.pkl\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "97ad9298-cede-4d31-b18a-a73a989683ea",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
