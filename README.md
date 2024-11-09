# 📧 Email Spam Detection with Machine Learning

Welcome to the **Email Spam Detection Project**! This project leverages machine learning to distinguish spam emails from legitimate emails (ham) using logistic regression. We preprocess email content, vectorize it, and train a model to accurately classify email messages. Let’s dive into the details! 🚀

## 🔍 Project Overview

This project reads a dataset of emails, processes and vectorizes the text, and trains a logistic regression model to detect spam. Using **CountVectorizer** and **Logistic Regression**, we classify emails with high accuracy, providing insights into the most significant features that separate spam from ham. 

## 🛠️ Project Structure

- **Data Loading**: Loads email data from a specified directory.
- **Preprocessing**: Cleans email content by removing non-alphabet characters and transforming to lowercase.
- **Vectorization**: Converts email content to a numerical format using `CountVectorizer`.
- **Model Training**: Trains a logistic regression classifier to detect spam.
- **Evaluation**: Measures the accuracy of the model and extracts key features indicative of spam.

## 📁 Code Structure

- `read_spam()`, `read_ham()`, `read_category()`: Load spam and ham emails from directories and process content into lists.
- `preprocessor(e)`: Preprocesses email content by removing non-alphabet characters and converting to lowercase.
- **Vectorization**: Uses `CountVectorizer` to transform emails into a word frequency matrix.
- **Logistic Regression**: Fits a logistic regression model and outputs training and test accuracies.
- **Feature Analysis**: Extracts and displays the top positive and negative features contributing to the model’s classification.

## 🧰 Prerequisites

- **Python 3.x** and libraries:
  - `pandas` for data handling
  - `sklearn` for machine learning and preprocessing
  - `re` for regex operations

## 🚀 Getting Started

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/email-spam-detection.git


📊 Results


After training the model, we achieve approximately 98% accuracy on the test data. The logistic regression model effectively identifies key words that correlate with spam or ham emails.



Model Evaluation Metrics:


Training Accuracy: ~99.97%  
Test Accuracy: ~98%  
Confusion Matrix: Provides insights into true positives, true negatives, false positives, and false negatives.  
Top Features: Displays top words for both spam and ham classifications.  


Sample Output:


plaintext  
Copy code  
Training accuracy: 0.9997  
Test accuracy: 0.98  


Confusion Matrix:  
[[732  17]
 [  8 278]]

 
Classification Report:

 
              precision    recall  f1-score   support

         ham       0.99      0.98      0.98       749
        spam       0.94      0.97      0.96       286

        
📈 Top 10 Spam-Indicating Features:


Positive (Spam): prices, http, pain, money, remove  
Negative (Ham): attached, enron, doc, thanks, meter    
These terms help distinguish spam emails (which often mention "prices," "remove," and "money") from ham (with terms like "attached," "thanks," and "enron").  



📬 Sample Usage  


To use the model on new emails:  


Preprocess the email content using preprocessor().  
Vectorize the preprocessed email with CountVectorizer.  


Predict using the trained logistic regression model:
python  
Copy code  
email_content = "Special offer just for you!"  
processed_content = preprocessor(email_content)  
vectorized_content = vectorizer.transform([processed_content])  
prediction = log_reg_model.predict(vectorized_content)  
print("Spam" if prediction[0] == "spam" else "Ham")  


📜 License  
This project is licensed under the MIT License.  



Happy coding, and may your inbox be spam-free! 📭✨ 

python  
Copy code  



This README provides all essential details, in an engaging and friendly format. Let me know if you’d like any changes!





