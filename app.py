import streamlit as st
import pickle

# Load model
model = pickle.load(open("D:\Netflix project\model\kmeans.pkl", "rb"))
tfidf = pickle.load(open("model/tfidf.pkl", "rb"))

st.title("Netflix Content Clustering")

user_input = st.text_input("Enter movie description")

if user_input:
    vec = tfidf.transform([user_input])
    cluster = model.predict(vec)

    st.write("Cluster:", cluster)