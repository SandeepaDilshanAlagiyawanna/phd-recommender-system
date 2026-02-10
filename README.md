## 🎓 AI-Based PhD Opportunity Recommendation System

This project implements a **content-based recommender system** that suggests **relevant UK PhD studentships** based on a user’s **research interests** using **Natural Language Processing (NLP)** and **Sentence-BERT embeddings**.

The system analyzes real PhD opportunity data and returns the **top matching studentships** ranked by **semantic similarity**.

---

## 📂 Dataset

**File name:** `data.csv`

The dataset contains UK PhD studentship opportunities scraped from:

```
https://www.jobs.ac.uk/phd
```

### Columns included

- **title** – Research topic of the PhD
- **employer** – University or institution offering the studentship
- **department** – Associated department or school
- **salary** – Annual stipend or funding information
- **location** – City of the PhD position
- **post_date** – Date the listing was posted
- **close_date** – Application deadline
- **url** – Direct link to the opportunity

---

## ⚙️ Methodology

This system uses a **content-based recommendation approach**:

1. **Text preprocessing**
   - Combine title, department, and university into a single description.

2. **Semantic embedding**
   - Convert descriptions into numerical vectors using
     **Sentence-BERT (`all-MiniLM-L6-v2`)**.

3. **Similarity computation**
   - Encode the user’s research interest.
   - Compute **cosine similarity** between the query and all PhD descriptions.

4. **Ranking**
   - Return the **Top-K most similar PhD studentships**.

---

## 🧠 Technologies Used

- **Python**
- **pandas** – data handling
- **sentence-transformers** – BERT embeddings
- **scikit-learn** – cosine similarity

---

## ▶️ How to Run

### 1. Install dependencies

```bash
pip install pandas sentence-transformers scikit-learn
```

### 2. Place dataset

Put the dataset in the project folder as:

```
data.csv
```

### 3. Run the recommender

```bash
python phd_recommender.py
```

### 4. Enter research interest

Example:

```
quantum computing machine learning
```

The system will output the **top recommended PhD studentships** with:

- Title
- University
- Department
- Location
- Salary
- Application link
- Similarity score

---

## 📊 Example Output

The recommender successfully returns **quantum-related PhD positions** when querying quantum computing topics, demonstrating correct **semantic matching and ranking**.

---

## 🎯 Project Significance

- Uses **real-world PhD opportunity data**
- Applies **modern NLP embeddings (BERT)**
- Implements a **true recommender system**
- Provides **practical academic career guidance**

This makes the project suitable for:

- **AI / Machine Learning coursework**
- **Recommender Systems modules**
- **Final-year undergraduate projects**

---

## 🔮 Future Improvements

- Add **filters** (location, salary, deadline)
- Build a **web interface (Streamlit)**
- Use **hybrid recommender (BERT + collaborative filtering)**
- Continuously **update dataset from jobs.ac.uk**
