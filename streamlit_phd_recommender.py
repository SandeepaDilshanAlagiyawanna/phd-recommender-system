import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime


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
        "post_date",
    ]

    df = df[required_cols]
    df = df.dropna(subset=["title"])

    # Convert post_date to datetime if available
    if "post_date" in df.columns:
        df["post_date"] = pd.to_datetime(df["post_date"], errors="coerce")

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
# Hybrid reranking components
# -----------------------------
def keyword_overlap_score(query, text):
    query_words = set(query.lower().split())
    text_words = set(text.lower().split())
    if not query_words:
        return 0
    return len(query_words & text_words) / len(query_words)


def recency_score(post_date):
    if pd.isna(post_date):
        return 0
    days_old = (datetime.now() - post_date).days
    return max(0, 1 - days_old / 365)  # newer posts score higher


# -----------------------------
# Hybrid Recommendation function
# -----------------------------
def recommend(query, model, embeddings, df, top_k=5, bert_w=0.7, kw_w=0.2, rec_w=0.1):

    # --- Stage 1: Neural retrieval ---
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]

    df_temp = df.copy()
    df_temp["bert_similarity"] = similarities

    # Take top 20 for reranking
    df_temp = df_temp.sort_values("bert_similarity", ascending=False).head(20)

    # --- Stage 2: Hybrid reranking ---
    combined_text = df_temp["title"].fillna("") + " " + df_temp["department"].fillna("")

    df_temp["keyword_score"] = [
        keyword_overlap_score(query, text) for text in combined_text
    ]

    if "post_date" in df_temp.columns:
        df_temp["recency_score"] = df_temp["post_date"].apply(recency_score)
    else:
        df_temp["recency_score"] = 0

    # Final hybrid score with customizable weights
    df_temp["final_score"] = (
        bert_w * df_temp["bert_similarity"]
        + kw_w * df_temp["keyword_score"]
        + rec_w * df_temp["recency_score"]
    )

    # Final ranking
    results = df_temp.sort_values("final_score", ascending=False).head(top_k)

    return results


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="PhD Recommender", page_icon="🎓", layout="wide")

st.title("🎓 Hybrid Neural PhD Recommender System")
st.write(
    "Find relevant UK PhD studentships using **BERT semantic search + hybrid reranking**, inspired by modern recommender systems."
)


# Load data
df = load_data("data.csv")
corpus = build_corpus(df)
model, embeddings = load_model_and_embeddings(corpus)

st.sidebar.header("⚙️ Settings")

top_k = st.sidebar.slider("Number of recommendations", 1, 10, 5)

# Filters
st.sidebar.subheader("🔍 Filters")
locations = ["All"] + sorted(df["location"].dropna().unique().tolist())
selected_location = st.sidebar.selectbox("Location", locations)

employers = ["All"] + sorted(df["employer"].dropna().unique().tolist())
selected_employer = st.sidebar.selectbox("University", employers)

# Minimum similarity threshold
min_score = st.sidebar.slider("Minimum match score", 0.0, 1.0, 0.0, 0.05)

# Sorting options
sort_by = st.sidebar.radio(
    "Sort results by", ["Match Score", "Title", "Location", "Recency"]
)

# Hybrid scoring weights
with st.sidebar.expander("⚖️ Adjust Scoring Weights"):
    st.caption("Customize how results are ranked:")
    bert_weight = st.slider("BERT Similarity", 0.0, 1.0, 0.7, 0.1)
    keyword_weight = st.slider("Keyword Match", 0.0, 1.0, 0.2, 0.1)
    recency_weight = st.slider("Recency", 0.0, 1.0, 0.1, 0.1)

    total_weight = bert_weight + keyword_weight + recency_weight
    if total_weight > 0:
        bert_weight /= total_weight
        keyword_weight /= total_weight
        recency_weight /= total_weight

# Use columns for better layout
col1, col2 = st.columns([3, 1])

with col1:
    query = st.text_input(
        "🔎 Enter your research interest:",
        placeholder="e.g., quantum computing, machine learning, robotics",
        help="Type your research interest and results will update automatically",
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
        height=100,
    )


# -----------------------------
# Explanation section
# -----------------------------

# Initialize session state for toggle
if "show_explanation" not in st.session_state:
    st.session_state.show_explanation = False

if st.button("ℹ️ Show How This Hybrid Recommender Works"):
    st.session_state.show_explanation = not st.session_state.show_explanation

if st.session_state.show_explanation:
    st.subheader("How the Recommendation System Works")

    st.markdown(
    """
### Stage 1 — Neural Content-Based Retrieval
Sentence-BERT converts PhD descriptions into **semantic embeddings** and retrieves
the **most relevant candidates using cosine similarity**.

This represents a **neural content-based recommender**, where recommendations are
generated from **textual similarity between user interests and item descriptions**.

### Stage 2 — Hybrid Re-Ranking Layer
The retrieved candidates are further refined using a **hybrid scoring mechanism**
that incorporates:

- **Neural semantic similarity (BERT)**
- **Keyword overlap relevance**
- **Recency of posting**

Final ranking score:

`final_score = 0.7 × BERT + 0.2 × Keyword + 0.1 × Recency`

This two-stage **retrieval → re-ranking pipeline** reflects the architecture used in
**modern industrial recommender systems and search engines**, moving beyond
traditional collaborative filtering toward **hybrid neural recommendation**.
    """
)


# -----------------------------
# Recommendation results
# -----------------------------
should_search = (enable_auto_search and query) or (
    not enable_auto_search and st.button("🎯 Find PhD Opportunities", type="primary")
)

if should_search and query:
    # Combine query with additional keywords
    full_query = query
    if additional_keywords:
        keywords_list = [
            k.strip() for k in additional_keywords.split("\n") if k.strip()
        ]
        if keywords_list:
            full_query = query + " " + " ".join(keywords_list)

    with st.spinner("🔍 Running hybrid neural search..."):
        results = recommend(
            full_query,
            model,
            embeddings,
            df,
            top_k * 2,
            bert_weight,
            keyword_weight,
            recency_weight,
        )

        # Apply filters
        if selected_location != "All":
            results = results[results["location"] == selected_location]

        if selected_employer != "All":
            results = results[results["employer"] == selected_employer]

        # Apply minimum score threshold
        results = results[results["final_score"] >= min_score]

        # Apply sorting
        if sort_by == "Title":
            results = results.sort_values("title")
        elif sort_by == "Location":
            results = results.sort_values("location")
        elif sort_by == "Recency" and "post_date" in results.columns:
            results = results.sort_values("post_date", ascending=False)
        else:  # Match Score is default
            results = results.sort_values("final_score", ascending=False)

        # Limit to top_k after filtering
        results = results.head(top_k)

    if len(results) == 0:
        st.warning(
            "🔍 No results found matching your criteria. Try adjusting the filters or search terms."
        )
    else:
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(
            ["📋 List View", "📊 Visual Analysis", "📈 Statistics"]
        )

        with tab1:
            st.subheader(f"🎓 Top {len(results)} Recommended PhD Studentships")
            st.caption(f"Found {len(results)} opportunities matching your interests")

            for idx, (_, row) in enumerate(results.iterrows(), 1):
                # Color code based on final score
                if row["final_score"] >= 0.7:
                    score_color = "🟢"
                elif row["final_score"] >= 0.5:
                    score_color = "🟡"
                else:
                    score_color = "🟠"

                with st.expander(
                    f"{score_color} **#{idx} - {row['title'][:80]}{'...' if len(row['title']) > 80 else ''}**",
                    expanded=(idx == 1),
                ):
                    col_a, col_b = st.columns([2, 1])

                    with col_a:
                        st.markdown(f"### {row['title']}")
                        st.markdown(f"**🏛️ University:** {row['employer']}")
                        st.markdown(f"**🔬 Department:** {row['department']}")
                        st.markdown(f"**📍 Location:** {row['location']}")
                        st.markdown(f"**💰 Salary:** {row['salary']}")
                        if "post_date" in row and not pd.isna(row["post_date"]):
                            st.markdown(
                                f"**📅 Posted:** {row['post_date'].strftime('%Y-%m-%d')}"
                            )

                    with col_b:
                        # Overall match score
                        st.metric(
                            "Match Score", f"{row['final_score']:.1%}", delta=None
                        )
                        st.progress(row["final_score"])

                        # Score breakdown
                        st.caption("**Score Breakdown:**")
                        st.caption(f"🤖 BERT: {row['bert_similarity']:.2f}")
                        st.caption(f"🔑 Keywords: {row['keyword_score']:.2f}")
                        st.caption(f"🕒 Recency: {row['recency_score']:.2f}")

                        st.markdown(f"[🔗 View Full Details]({row['url']})")

                    st.divider()

        with tab2:
            st.subheader("📊 Match Score Distribution")

            # Create final score chart
            fig = go.Figure(
                go.Bar(
                    x=[
                        f"{i+1}. {row['title'][:30]}..."
                        for i, (_, row) in enumerate(results.iterrows())
                    ],
                    y=results["final_score"].values,
                    marker=dict(
                        color=results["final_score"].values,
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title="Match Score"),
                    ),
                    text=[f"{s:.1%}" for s in results["final_score"].values],
                    textposition="auto",
                )
            )

            fig.update_layout(
                xaxis_title="PhD Opportunity",
                yaxis_title="Final Score",
                yaxis_range=[0, 1],
                height=400,
                showlegend=False,
            )

            st.plotly_chart(fig, use_container_width=True)

            # Score component breakdown
            st.subheader("⚖️ Score Component Analysis")
            score_components = pd.DataFrame(
                {
                    "BERT Similarity": results["bert_similarity"].values,
                    "Keyword Match": results["keyword_score"].values,
                    "Recency": results["recency_score"].values,
                },
                index=[
                    f"{i+1}. {row['title'][:25]}..."
                    for i, (_, row) in enumerate(results.iterrows())
                ],
            )

            fig2 = go.Figure()
            for column in score_components.columns:
                fig2.add_trace(
                    go.Bar(
                        name=column,
                        x=score_components.index,
                        y=score_components[column],
                    )
                )

            fig2.update_layout(
                barmode="group",
                yaxis_title="Score",
                xaxis_title="PhD Opportunity",
                height=400,
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
            )

            st.plotly_chart(fig2, use_container_width=True)

            # Location distribution
            if len(results["location"].unique()) > 1:
                st.subheader("📍 Opportunities by Location")
                location_counts = results["location"].value_counts()
                fig3 = px.pie(
                    values=location_counts.values, names=location_counts.index, hole=0.4
                )
                fig3.update_layout(height=400)
                st.plotly_chart(fig3, use_container_width=True)

        with tab3:
            st.subheader("📈 Quick Statistics")

            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

            with col_stat1:
                st.metric("Total Results", len(results))

            with col_stat2:
                st.metric("Avg Match Score", f"{results['final_score'].mean():.1%}")

            with col_stat3:
                st.metric("Best Match", f"{results['final_score'].max():.1%}")

            with col_stat4:
                st.metric("Unique Universities", results["employer"].nunique())

            st.divider()

            # Average score components
            st.subheader("⚖️ Average Score Components")
            col_avg1, col_avg2, col_avg3 = st.columns(3)

            with col_avg1:
                st.metric("Avg BERT Score", f"{results['bert_similarity'].mean():.3f}")

            with col_avg2:
                st.metric("Avg Keyword Match", f"{results['keyword_score'].mean():.3f}")

            with col_avg3:
                st.metric("Avg Recency Score", f"{results['recency_score'].mean():.3f}")

            st.divider()

            # Top universities
            st.subheader("🏆 Top Universities in Results")
            top_unis = results["employer"].value_counts().head(5)
            for uni, count in top_unis.items():
                st.write(f"**{uni}** - {count} opportunity(ies)")

            # Recent postings if available
            if "post_date" in results.columns and not results["post_date"].isna().all():
                st.divider()
                st.subheader("📅 Most Recent Postings")
                recent = results.sort_values("post_date", ascending=False).head(3)
                for _, r in recent.iterrows():
                    if not pd.isna(r["post_date"]):
                        st.write(
                            f"**{r['title'][:60]}{'...' if len(r['title']) > 60 else ''}** - {r['post_date'].strftime('%Y-%m-%d')}"
                        )

elif not query:
    st.info(
        "👋 Welcome! Enter a research topic above to see personalized PhD recommendations."
    )

    # Show some sample queries
    with st.expander("💡 Need inspiration? Try these sample searches:"):
        col_sample1, col_sample2, col_sample3 = st.columns(3)

        with col_sample1:
            st.markdown(
                """**Computer Science:**
- Machine Learning
- Cybersecurity
- Quantum Computing"""
            )

        with col_sample2:
            st.markdown(
                """**Life Sciences:**
- Cancer Research
- Genomics
- Neuroscience"""
            )

        with col_sample3:
            st.markdown(
                """**Engineering:**
- Renewable Energy
- Robotics
- Materials Science"""
            )

else:
    st.info("Click the **Find PhD Opportunities** button to search.")
