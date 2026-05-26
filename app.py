import streamlit as st
import pickle
from groq import Groq
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Load model
model = pickle.load(open("kmeans.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

st.title("🎬 Netflix Content Clustering & Recommendation")

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
    cluster_names = {
    0: "Drama",
    1: "Comedy",
    2: "Action",
    3: "Romance",
    4: "Crime",
    5: "Documentary"}
    st.success(f"Cluster: {cluster_names.get(int(cluster[0]))}")


# load data
df = pd.read_csv("NETFLIX MOVIES AND TV SHOWS CLUSTERING.csv")

# transform full dataset
X = tfidf.transform(df['description'])

# user input
vec = tfidf.transform([user_input])

# similarity
similarity = cosine_similarity(vec, X)

# top 5
top_idx = similarity.argsort()[0][-5:]

st.write("🎬 Similar Content:")
for i in top_idx:
    st.write(df.iloc[i]['title'])
