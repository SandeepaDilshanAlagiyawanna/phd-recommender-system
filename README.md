## 🎓 Hybrid Neural PhD Opportunity Recommendation System

This project implements a **modern AI-driven recommender system** that suggests
**relevant UK PhD studentships** based on a user’s **research interests**.

Unlike traditional recommenders that rely only on **collaborative filtering**,
this system adopts a **hybrid neural recommendation architecture** combining:

* **Transformer-based semantic understanding (Sentence-BERT)**
* **Content-based similarity retrieval**
* **Hybrid re-ranking inspired by learning-to-rank recommender pipelines**

to generate **personalized and explainable PhD recommendations**.

---

## 📂 Dataset

**File name:** `data.csv`

Source:

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
* **url** – Direct application link

---

## 🧠 Recommender System Architecture

This project follows a **two-stage hybrid neural recommendation pipeline**
commonly used in **modern industrial recommender systems**.

### Stage 1 — Neural Content-Based Retrieval

* Text fields (**title, department, employer**) are merged into a single description.
* Descriptions are encoded into **semantic embeddings** using:

```
Sentence-BERT (all-MiniLM-L6-v2)
```

* The user query is embedded and matched via **cosine similarity**
  to retrieve the **top candidate PhD opportunities**.

This stage represents a **neural content-based recommender**,
capable of understanding **semantic meaning beyond keywords**.

---

### Stage 2 — Hybrid Re-Ranking (Learning-to-Rank Inspired)

Retrieved candidates are refined using a **hybrid scoring mechanism**:

* **Neural semantic similarity (BERT score)**
* **Keyword overlap relevance**
* **Recency of posting**

Final ranking score:

```
final_score =
  0.7 × BERT similarity
+ 0.2 × keyword relevance
+ 0.1 × recency score
```

This **retrieval → re-ranking architecture** reflects the design used in:

* Search engines
* Industrial recommender systems
* Modern recommender-system research

and moves **beyond traditional collaborative filtering**.

---

## ⚙️ Technologies Used

* **Python**
* **Streamlit** – Interactive web interface
* **pandas** – Data preprocessing
* **sentence-transformers** – Transformer embeddings
* **scikit-learn** – Cosine similarity computation
* **Plotly** – Interactive visualization

---

## ▶️ Running the Application Locally

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place dataset

Ensure the dataset file exists as:

```
data.csv
```

### 3. Launch Streamlit app

```bash
streamlit run streamlit_phd_recommender.py
```

---

## 🌐 Deployment

The application can be deployed using:

* **GitHub** – Source code hosting
* **Streamlit Cloud** – Public interactive web application

Users can:

* Enter research interests
* View **ranked PhD recommendations**
* Analyze **match scores and statistics**
* Access **direct application links**

---

## 🎯 Key Contributions

This project demonstrates:

* Application of **transformer-based NLP in recommender systems**
* Design of a **hybrid neural retrieval + re-ranking pipeline**
* Implementation of **explainable recommendation scoring**
* Practical deployment of an **interactive AI recommender web app**

Suitable for:

* **Machine Learning coursework**
* **Recommender Systems modules**
* **Final-year undergraduate AI projects**

---

## 🔮 Future Work

* Integrate **true collaborative filtering** using user interaction history
* Apply **learning-to-rank neural models**
* Add **context-aware or conversational recommendation (RAG-based)**
* Enable **automatic dataset updates from jobs.ac.uk**

---

## 📜 License

This project is developed for **academic and educational purposes**.