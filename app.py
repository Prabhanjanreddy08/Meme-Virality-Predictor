import streamlit as st
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import shap

# Load dataset
df = pd.read_csv("meme_virality_dataset.csv")

# Preprocessing
df['caption_sentiment'] = df['caption'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['caption_length'] = df['caption'].apply(len)
le = LabelEncoder()
df['platform_encoded'] = le.fit_transform(df['platform'])

# Feature setup
features = ['likes', 'comments', 'shares', 'time_posted_hour', 'caption_sentiment', 'caption_length', 'platform_encoded']
X = df[features]
y = df['viral']

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# App Title
st.title("🔥 Meme Virality Predictor")
st.write("Will your meme go viral? Drop some early stats and find out.")

# Input Section
with st.sidebar:
    st.header("Input Meme Metrics")
    platform = st.selectbox("Platform", ["Reddit", "Twitter", "Instagram"])
    caption = st.text_area("Caption", "When your code works on the first try")
    likes = st.slider("Likes", 0, 1000000, 100)
    comments = st.slider("Comments", 0, 10000, 50)
    shares = st.slider("Shares", 0, 10000, 100)
    time_posted_hour = st.slider("Posted Hour (0–24)", 0, 23, 15)

# Feature extraction
caption_sentiment = TextBlob(caption).sentiment.polarity
caption_length = len(caption)
platform_encoded = le.transform([platform])[0]

input_features = pd.DataFrame([{
    'likes': likes,
    'comments': comments,
    'shares': shares,
    'time_posted_hour': time_posted_hour,
    'caption_sentiment': caption_sentiment,
    'caption_length': caption_length,
    'platform_encoded': platform_encoded
}])

explainer = shap.Explainer(model, X)

# Predict
if st.button("Predict"):
    prediction = model.predict(input_features)[0]
    prob = model.predict_proba(input_features)[0][1]
    
    if prediction == 1:
        st.success(f"🔥 Viral! (Probability: {prob:.2%})")
    else:
        st.warning(f"🙃 Not viral... yet. (Probability: {prob:.2%})")
    
# Get input values and feature importances
input_vals = input_features.iloc[0].values
importances = model.feature_importances_

# Element-wise multiplication (value × importance)
contribution = input_vals * importances

# Clip anything below 0 (just to be safe)
contribution = np.clip(contribution, 0, None)

# Build DataFrame
contrib_df = pd.DataFrame({
    'Feature': features,
    'Influence': contribution
}).sort_values(by='Influence', ascending=False)

# Plot
st.subheader("📊 Feature Influence on Prediction")
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x='Influence', y='Feature', data=contrib_df, ax=ax, palette="Blues_d")
st.pyplot(fig)


# Feature importance
st.subheader("📊 Feature Importance")
importances = pd.Series(model.feature_importances_, index=features).sort_values()
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x=importances.values, y=importances.index, ax=ax)
st.pyplot(fig)