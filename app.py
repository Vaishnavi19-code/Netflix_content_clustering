import streamlit as st
import pickle
from groq import Groq
# Load model
model = pickle.load(open("kmeans.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

st.title("Netflix Content Clustering")

user_input = st.text_input("Enter movie/show name or description")
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

def explain(text):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": f"Explain this Netflix content: {text}"}
        ]
    )
    return response.choices[0].message.content

if user_input:
    vec = tfidf.transform([user_input])
    cluster = model.predict(vec)

    explanation = explain(user_input)

    st.write("Explanation:", explanation)
    st.subheader("🎯 Cluster Result")
    st.success(f"Cluster: {int(cluster[0])}")
