import streamlit as st
import pickle
from groq import Groq
# Load model
model = pickle.load(open("kmeans.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

st.title("Netflix Content Clustering")

user_input = st.text_input("Enter movie description")
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

def explain(text):
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "user", "content": f"Explain this Netflix content: {text}"}
        ]
    )
    return response.choices[0].message.content

if user_input:
    vec = tfidf.transform([user_input])
    cluster = model.predict(vec)

    explanation = explain(user_input)

    st.write("Cluster:", cluster)
    st.write("Explanation:", explanation)

