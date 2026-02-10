## 🎓 AI-Based PhD Opportunity Recommendation System

### (Using NLP + Collaborative Filtering Concepts)

This project implements an **AI-driven recommender system** that suggests **relevant UK PhD studentships** based on a user’s **research interests**.

The system combines:

* **Semantic text understanding using Sentence-BERT**
* **Similarity ranking inspired by Collaborative Filtering principles**

to produce **personalized PhD recommendations**.

---

## 📂 Dataset

**File name:** `data.csv`

The dataset contains real UK PhD studentship opportunities scraped from:

```
https://www.kaggle.com/datasets/xiangyiz/uk-phd-studentship
```

### Dataset fields

* **title** – Research topic of the PhD
* **employer** – University offering the studentship
* **department** – Academic department
* **salary** – Funding / stipend information
* **location** – City of the position
* **post_date** – Date posted
* **close_date** – Application deadline
* **url** – Link to apply

---

## 🧠 Recommender System Approach

This project follows a **content-based recommendation pipeline** enhanced with
**Collaborative Filtering intuition**.

### Step 1 — Text Representation

Relevant text fields (title, department, employer) are merged into a single description.

### Step 2 — Semantic Embedding

Descriptions are converted into numerical vectors using:

```
Sentence-BERT (all-MiniLM-L6-v2)
```

### Step 3 — Similarity Matching

The user’s research query is embedded and compared with all PhD vectors using:

```
Cosine similarity
```

### Step 4 — Ranking (Collaborative Filtering Insight)

Opportunities with the **highest similarity scores** are ranked and returned,
mirroring the **core idea of collaborative filtering**:

> *Items similar to user preferences are recommended.*

---

## ⚙️ Technologies Used

* **Python**
* **Streamlit** – Web interface
* **pandas** – Data processing
* **sentence-transformers** – BERT embeddings
* **scikit-learn** – Similarity computation

---

## ▶️ How to Run Locally

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Ensure dataset exists

```
data.csv
```

### 3. Run Streamlit app

```bash
streamlit run streamlit_phd_recommender.py
```

---

## 🌐 Live Deployment

The application can be deployed on:

* **GitHub** (code hosting)
* **Streamlit Cloud** (public web app)

After deployment, users can:

* Enter research interests
* Receive **ranked PhD recommendations**
* Access **direct application links**

---

## 🎯 Project Contribution

This project demonstrates:

* Practical use of **NLP in recommender systems**
* Application of **semantic similarity with BERT**
* Recommendation logic aligned with
  **Collaborative Filtering principles used in modern AI systems**

Suitable for:

* **Machine Learning coursework**
* **Recommender Systems modules**
* **Final-year undergraduate projects**

---

## 🔮 Future Work

* Add **true hybrid collaborative filtering** with user interaction history
* Implement **advanced ranking models**
* Provide **filters for salary, location, and deadline**
* Enable **automatic dataset updates**
