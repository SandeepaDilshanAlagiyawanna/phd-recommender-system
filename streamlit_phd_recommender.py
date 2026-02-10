import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# Load and clean dataset
# -----------------------------
@st.cache_data
def load_data(path="data.csv"):
    df = pd.read_csv(path)

    df.columns = df.columns.str.strip().str.lower()

    required_cols = [
        "title",
        "employer",
        "department",
        "salary",
        "location",
        "url",
    ]

    df = df[required_cols]
    df = df.dropna(subset=["title"])
    df = df.reset_index(drop=True)

    return df


# -----------------------------
# Build corpus for embeddings
# -----------------------------
def build_corpus(df):
    corpus = (
        df["title"].fillna("")
        + " "
        + df["department"].fillna("")
        + " "
        + df["employer"].fillna("")
    )
    return corpus.tolist()


# -----------------------------
# Load model & embeddings
# -----------------------------
@st.cache_resource
def load_model_and_embeddings(corpus):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(corpus, show_progress_bar=False)
    return model, embeddings


# -----------------------------
# Recommendation function
# -----------------------------
def recommend(query, model, embeddings, df, top_k=5):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]

    top_indices = similarities.argsort()[-top_k:][::-1]

    results = df.iloc[top_indices].copy()
    results["similarity"] = similarities[top_indices]

    return results


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="PhD Recommender", page_icon="🎓", layout="wide")

st.title("🎓 AI-Based PhD Opportunity Recommender")
st.write(
    "Enter your **research interests** to find the most relevant UK PhD studentships using NLP and BERT embeddings."
)


# Load data
df = load_data("data.csv")
corpus = build_corpus(df)
model, embeddings = load_model_and_embeddings(corpus)

st.sidebar.header("Settings")

top_k = st.sidebar.slider("Number of recommendations", 1, 10, 5)


query = st.text_input(
    "Enter your research interest:",
    placeholder="e.g., quantum computing, machine learning, robotics",
)


# -----------------------------
# Explanation section button
# -----------------------------
if st.button("ℹ️ Show How This Recommender Works"):
    st.subheader("How the Recommendation System Works")

    st.markdown(
        """
### 1️⃣ Text Processing
The system combines **PhD title, department, and university** into a single text description.

### 2️⃣ Semantic Embedding (BERT)
Each description is converted into a **numerical vector** using the
**Sentence‑BERT model (`all-MiniLM-L6-v2`)**.

This allows the system to understand **meaning**, not just keywords.

### 3️⃣ Similarity Matching
Your research interest is also converted into a vector, and the system
computes **cosine similarity** between your query and every PhD opportunity.

### 4️⃣ Ranking (Collaborative Filtering Insight)
The PhD positions with the **highest similarity scores** are ranked and returned,
following the **core intuition of collaborative filtering**:

> Recommend items that are most similar to the user’s preferences.

---

✅ This makes the project a **content‑based recommender system enhanced with
collaborative filtering principles**, commonly used in modern AI applications.
        """
    )


# -----------------------------
# Recommendation results
# -----------------------------
if st.button("Find PhD Opportunities") and query:
    results = recommend(query, model, embeddings, df, top_k)

    st.subheader("Top Recommended PhD Studentships")

    for _, row in results.iterrows():
        with st.container():
            st.markdown(f"### {row['title']}")
            st.write(f"**University:** {row['employer']}")
            st.write(f"**Department:** {row['department']}")
            st.write(f"**Location:** {row['location']}")
            st.write(f"**Salary:** {row['salary']}")
            st.write(f"**Match Score:** {row['similarity']:.3f}")
            st.markdown(f"[🔗 View Details]({row['url']})")
            st.divider()

else:
    st.info("Enter a research topic and click **Find PhD Opportunities** to see recommendations.")