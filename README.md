# 🔥 Meme Virality Predictor

#[Visit the Meme Virality Predictor](https://meme-virality-predictor-ih7xc4ccnbazdvdcwxvjrc.streamlit.app/)


This is a Streamlit web application that predicts the likelihood of a meme going viral based on user-provided metrics such as likes, comments, platform, caption sentiment, and more. The app uses a Random Forest Classifier to make predictions and visualizes both prediction influence and overall feature importance.

---

## 📦 Features

- Input meme stats via an interactive sidebar
- Sentiment analysis of captions using **TextBlob**
- Random Forest-based virality prediction
- Probability-based feedback on potential virality
- Feature influence breakdown for your specific input
- Global feature importance from the trained model
- SHAP explainer initialized for potential interpretability

---

## 🚀 How to Run the App

### Prerequisites

Install required packages:

    pip install streamlit pandas numpy textblob scikit-learn seaborn matplotlib shap

    streamlit run app.py
# How It Works
1.Data Loading and Preprocessing:

    Loads meme data from meme_virality_dataset.csv.
    Extracts sentiment polarity from meme captions using TextBlob.
    Computes caption length.
    Encodes platform types (e.g., Reddit, Twitter, Instagram).

Model Training:

    A RandomForestClassifier is trained on features:
    Likes
    Comments
    Shares
    Time posted (hour)
    Caption sentiment
    Caption length
    Encoded platform

User Input:

    Sidebar collects meme details.
    Sentiment and caption length are auto-generated.

Prediction:
    
    Displays whether the meme is likely to go viral or not.
    Shows the probability of virality.
    Visualizes feature influence for your input.
    Displays global feature importance.


# 📊 Example Inputs
Try this test input in the app:

Platform: Instagram
Caption: "When your code works on the first try"
Likes: 100
Comments: 50
Shares: 100
Posted Hour: 15
