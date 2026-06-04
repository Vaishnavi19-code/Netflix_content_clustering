Overview
This project builds a content-based recommendation system using Netflix movie data. It leverages NLP, clustering techniques, and Generative AI to provide personalized movie recommendations along with contextual explanations.

Features
1. Content-based movie recommendations
2. NLP-based text preprocessing using NLTK
3. Clustering using K-Means & Hierarchical methods
4. Optimal cluster selection using Silhouette Score & Dendrograms
5. GenAI integration for dynamic movie insights
6. Interactive UI built with Streamlit

Tech Stack
Programming: Python
NLP: NLTK
Machine Learning: Scikit-learn
Techniques: TF-IDF, K-Means, Hierarchical Clustering
Evaluation: Silhouette Score, Dendrograms
Frontend: Streamlit
AI Integration: Generative AI (LLMs)

Methodology
1. Data Preprocessing
   - Cleaned and processed text data using NLTK
   - Removed stopwords, tokenized, and normalized text

2. Feature Engineering
   - Converted text into numerical form using TF-IDF

3. Clustering
   - Applied K-Means and Hierarchical Clustering
   - Used Silhouette Score & Dendrograms for evaluation
   - Selected optimal clusters for recommendations

4. Recommendation System
   - Identified similar movies based on cluster grouping

5. GenAI Integration
   - Generated contextual explanations and summaries for recommendations

Demo
- Input a movie name
- Get similar movie recommendations
- View AI-generated explanations

