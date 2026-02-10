import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go


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

st.sidebar.header("⚙️ Settings")

top_k = st.sidebar.slider("Number of recommendations", 1, 20, 5)

# Filters
st.sidebar.subheader("🔍 Filters")
locations = ["All"] + sorted(df["location"].dropna().unique().tolist())
selected_location = st.sidebar.selectbox("Location", locations)

employers = ["All"] + sorted(df["employer"].dropna().unique().tolist())
selected_employer = st.sidebar.selectbox("University", employers)

# Minimum similarity threshold
min_similarity = st.sidebar.slider("Minimum match score", 0.0, 1.0, 0.0, 0.05)

# Sorting options
sort_by = st.sidebar.radio("Sort results by", ["Similarity", "Title", "Location"])


# Use columns for better layout
col1, col2 = st.columns([3, 1])

with col1:
    query = st.text_input(
        "🔎 Enter your research interest:",
        placeholder="e.g., quantum computing, machine learning, robotics",
        help="Type your research interest and results will update automatically"
    )

with col2:
    st.write("")
    st.write("")
    enable_auto_search = st.checkbox("Auto-search", value=False)

# Multi-select for keywords
with st.expander("➕ Add multiple research interests (optional)"):
    additional_keywords = st.text_area(
        "Enter additional keywords (one per line):",
        placeholder="machine learning\nartificial intelligence\ndata science",
        height=100
    )


# -----------------------------
# Explanation section button
# -----------------------------
# Initialize session state for toggle
if "show_explanation" not in st.session_state:
    st.session_state.show_explanation = False

if st.button("ℹ️ Show How This Recommender Works"):
    st.session_state.show_explanation = not st.session_state.show_explanation

if st.session_state.show_explanation:
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
        """
    )


# -----------------------------
# Recommendation results
# -----------------------------
should_search = (enable_auto_search and query) or (not enable_auto_search and st.button("🎯 Find PhD Opportunities", type="primary"))

if should_search and query:
    # Combine query with additional keywords
    full_query = query
    if additional_keywords:
        keywords_list = [k.strip() for k in additional_keywords.split("\n") if k.strip()]
        if keywords_list:
            full_query = query + " " + " ".join(keywords_list)
    
    with st.spinner("🔍 Finding the best matches for you..."):
        results = recommend(full_query, model, embeddings, df, top_k * 2)  # Get more results for filtering
        
        # Apply filters
        if selected_location != "All":
            results = results[results["location"] == selected_location]
        
        if selected_employer != "All":
            results = results[results["employer"] == selected_employer]
        
        # Apply minimum similarity threshold
        results = results[results["similarity"] >= min_similarity]
        
        # Apply sorting
        if sort_by == "Title":
            results = results.sort_values("title")
        elif sort_by == "Location":
            results = results.sort_values("location")
        else:  # Similarity is default
            results = results.sort_values("similarity", ascending=False)
        
        # Limit to top_k after filtering
        results = results.head(top_k)
    
    if len(results) == 0:
        st.warning("🔍 No results found matching your criteria. Try adjusting the filters or search terms.")
    else:
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📋 List View", "📊 Visual Analysis", "📈 Statistics"])
        
        with tab1:
            st.subheader(f"🎓 Top {len(results)} Recommended PhD Studentships")
            st.caption(f"Found {len(results)} opportunities matching your interests")
            
            for idx, (_, row) in enumerate(results.iterrows(), 1):
                # Color code based on similarity
                if row['similarity'] >= 0.7:
                    similarity_color = "🟢"
                elif row['similarity'] >= 0.5:
                    similarity_color = "🟡"
                else:
                    similarity_color = "🟠"
                
                with st.expander(f"{similarity_color} **#{idx} - {row['title'][:80]}{'...' if len(row['title']) > 80 else ''}**", expanded=(idx == 1)):
                    col_a, col_b = st.columns([2, 1])
                    
                    with col_a:
                        st.markdown(f"### {row['title']}")
                        st.markdown(f"**🏛️ University:** {row['employer']}")
                        st.markdown(f"**🔬 Department:** {row['department']}")
                        st.markdown(f"**📍 Location:** {row['location']}")
                        st.markdown(f"**💰 Salary:** {row['salary']}")
                    
                    with col_b:
                        # Similarity gauge
                        st.metric("Match Score", f"{row['similarity']:.1%}", delta=None)
                        
                        # Visual similarity bar
                        st.progress(row['similarity'])
                        
                        st.markdown(f"[🔗 View Full Details]({row['url']})")
                    
                    st.divider()
        
        with tab2:
            st.subheader("📊 Match Score Distribution")
            
            # Create similarity score chart
            fig = go.Figure(go.Bar(
                x=[f"{i+1}. {row['title'][:30]}..." for i, (_, row) in enumerate(results.iterrows())],
                y=results['similarity'].values,
                marker=dict(
                    color=results['similarity'].values,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Match Score")
                ),
                text=[f"{s:.1%}" for s in results['similarity'].values],
                textposition='auto',
            ))
            
            fig.update_layout(
                xaxis_title="PhD Opportunity",
                yaxis_title="Similarity Score",
                yaxis_range=[0, 1],
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Location distribution
            if len(results['location'].unique()) > 1:
                st.subheader("📍 Opportunities by Location")
                location_counts = results['location'].value_counts()
                fig2 = px.pie(values=location_counts.values, names=location_counts.index, hole=0.4)
                fig2.update_layout(height=400)
                st.plotly_chart(fig2, use_container_width=True)
        
        with tab3:
            st.subheader("📈 Quick Statistics")
            
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            
            with col_stat1:
                st.metric("Total Results", len(results))
            
            with col_stat2:
                st.metric("Avg Match Score", f"{results['similarity'].mean():.1%}")
            
            with col_stat3:
                st.metric("Best Match", f"{results['similarity'].max():.1%}")
            
            with col_stat4:
                st.metric("Unique Universities", results['employer'].nunique())
            
            st.divider()
            
            # Top universities
            st.subheader("🏆 Top Universities in Results")
            top_unis = results['employer'].value_counts().head(5)
            for uni, count in top_unis.items():
                st.write(f"**{uni}** - {count} opportunity(ies)")

elif not query:
    st.info("👋 Welcome! Enter a research topic above to see personalized PhD recommendations.")
    
    # Show some sample queries
    with st.expander("💡 Need inspiration? Try these sample searches:"):
        col_sample1, col_sample2, col_sample3 = st.columns(3)
        
        with col_sample1:
            st.markdown("""**Computer Science:**
- Machine Learning
- Cybersecurity
- Quantum Computing""")
        
        with col_sample2:
            st.markdown("""**Life Sciences:**
- Cancer Research
- Genomics
- Neuroscience""")
        
        with col_sample3:
            st.markdown("""**Engineering:**
- Renewable Energy
- Robotics
- Materials Science""")

else:
    st.info("Click the **Find PhD Opportunities** button to search.")