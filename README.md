# Perfume Recommender

A hybrid machine learning web application that recommends perfumes based on fragrance notes or similar perfumes.

Live demo: [perfume-recommender.streamlit.app](https://perfume-recommender-hnigwzxqbvfcjurzkjhebk.streamlit.app)

---

## Overview

This project covers the full data science pipeline — from data collection to a deployed web application. The dataset was built by scraping [Parfumo](https://www.parfumo.com), a fragrance community website, collecting perfume names, brands, ratings, genres, and fragrance notes (top, heart, base). The raw data was then cleaned and preprocessed before being used to train the recommendation models.

The recommender uses a hybrid approach combining TF-IDF vectorization and KMeans clustering to suggest perfumes that match a user's scent preferences.

---

## Features

- Search by name — enter a perfume you like and get similar recommendations
- Search by notes — select fragrance notes (Vanilla, Oud, Bergamot, etc.) and get personalized suggestions
- Same cluster filter — restrict recommendations to the same scent family
- Genre filter — filter by Men, Women, or Unisex
- Scent families — perfumes grouped into 10 fragrance clusters with interpretable labels

---

## How It Works

```
User inputs a perfume name or selects notes
                    |
        TF-IDF vectorizer transforms notes into a vector
                    |
        Cosine Similarity finds the closest perfumes
                    |
        KMeans cluster adds scent family context
                    |
        Results ranked by similarity score
```

The hybrid design means the model does not rely solely on cosine similarity. KMeans clustering groups perfumes into scent families (e.g. warm oriental, fresh aquatic, soft floral), which can be used to filter recommendations and provide additional context to the user.

---

## Data Collection and Preparation

- Scraped over 1,500 perfumes from Parfumo using Python (Selenium, BeautifulSoup)
- Collected: name, brand, rating, vote count, genre, top notes, heart notes, base notes
- Cleaned missing values, normalized note formatting, and removed duplicates
- Final dataset used for modeling: 1,015 perfumes with complete note information

---

## Project Structure

```
perfume-recommender/
|
|-- data/
|   |-- perfume_dataset_backup.csv   # 1,015 perfumes with notes
|   |-- perfumes.csv                 # full dataset with ratings
|   |-- similarity_matrix.pkl        # cosine similarity matrix (1015x1015)
|   |-- tfidf_matrix.pkl             # TF-IDF matrix (1015x760)
|   |-- tfidf_vectorizer.pkl         # fitted TF-IDF vectorizer
|   `-- kmeans_model.pkl             # KMeans model (k=10)
|
|-- models/
|   |-- recommender.py               # model loading and recommendation logic
|   `-- clustering.py                # cluster analytics and labels
|
|-- app/
|   `-- app.py                       # Streamlit UI
|
|-- requirements.txt
`-- README.md
```

---

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

---

## Dataset Summary

- 1,015 perfumes with complete top, heart, and base notes
- 760 unique fragrance ingredients in the TF-IDF vocabulary
- 10 KMeans clusters representing distinct scent families
- Ratings and genre labels included

---

## Tech Stack

- Python 3.11
- Streamlit
- scikit-learn (TF-IDF, KMeans, Cosine Similarity)
- pandas, numpy
- joblib
- requests, BeautifulSoup (data collection)
- Selenium (dynamic page rendering during scraping)
