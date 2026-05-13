import ast
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def load_data():
    df = pd.read_csv(DATA_DIR / "perfume_dataset_backup.csv")
    df = df.head(1015).reset_index(drop=True)
    return df


def load_models():
    sim_matrix = joblib.load(DATA_DIR / "similarity_matrix.pkl")
    tfidf_mat = joblib.load(DATA_DIR / "tfidf_matrix.pkl")
    tfidf_vec = joblib.load(DATA_DIR / "tfidf_vectorizer.pkl")
    kmeans = joblib.load(DATA_DIR / "kmeans_model.pkl")
    return sim_matrix, tfidf_mat, tfidf_vec, kmeans


SKIP_LABELS = {"Top Notes", "Heart Notes", "Base Notes"}


def parse_notes(note_str: str) -> list[str]:
    try:
        lst = ast.literal_eval(note_str)
        return [n.strip() for n in lst if n.strip() not in SKIP_LABELS]
    except Exception:
        return []


def get_all_notes(row: pd.Series) -> list[str]:
    return (
            parse_notes(row["top_notes"])
            + parse_notes(row["middle_notes"])
            + parse_notes(row["base_notes"])
    )


def get_unique_notes(df: pd.DataFrame) -> list[str]:
    all_notes = set()
    for _, row in df.iterrows():
        all_notes.update(get_all_notes(row))
    return sorted(all_notes)


def recommend_by_name(
        perfume_name: str,
        df: pd.DataFrame,
        sim_matrix: np.ndarray,
        kmeans,
        top_n: int = 8,
        same_cluster_only: bool = False,
) -> pd.DataFrame:

    mask = df["name"].str.lower() == perfume_name.strip().lower()
    if not mask.any():
        mask = df["name"].str.lower().str.contains(perfume_name.strip().lower(), na=False)
        if not mask.any():
            return pd.DataFrame()

    idx = df[mask].index[0]
    cluster_id = kmeans.labels_[idx]


    scores = sim_matrix[idx].copy()
    scores[idx] = -1

    if same_cluster_only:
        cluster_mask = kmeans.labels_ == cluster_id
        non_cluster = ~cluster_mask
        scores[non_cluster] = -1

    top_indices = np.argsort(scores)[::-1][:top_n]

    results = df.loc[top_indices].copy()
    results["similarity_score"] = scores[top_indices]
    results["cluster"] = kmeans.labels_[top_indices]
    results["all_notes"] = results.apply(
        lambda r: ", ".join(get_all_notes(r)), axis=1
    )
    return results[["name", "brand", "rating", "genre", "similarity_score", "cluster", "all_notes"]]


def recommend_by_notes(
        selected_notes: list[str],
        df: pd.DataFrame,
        tfidf_mat,
        tfidf_vec,
        kmeans,
        top_n: int = 8,
        genre_filter: str | None = None,
) -> pd.DataFrame:

    if not selected_notes:
        return pd.DataFrame()

    query = " ".join(selected_notes).lower()
    query_vec = tfidf_vec.transform([query])
    scores = cosine_similarity(query_vec, tfidf_mat).flatten()

    results = df.copy()
    results["similarity_score"] = scores
    results["cluster"] = kmeans.labels_

    if genre_filter and genre_filter != "All":
        results = results[results["genre"] == genre_filter]

    results = results.sort_values("similarity_score", ascending=False).head(top_n)
    results["all_notes"] = results.apply(lambda r: ", ".join(get_all_notes(r)), axis=1)

    return results[["name", "brand", "rating", "genre", "similarity_score", "cluster", "all_notes"]]