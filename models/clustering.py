import ast
from collections import Counter

import numpy as np
import pandas as pd

CLUSTER_LABELS = {
    0: "Floral & Woody",
    1: "Fresh & Versatile",
    2: "Green & Earthy",
    3: "Warm Oriental",
    4: "Floral & Sweet",
    5: "Elegant & Refined",
    6: "Rich Floral",
    7: "Soft & Creamy",
    8: "Intimate & Musky",
    9: "Deep & Smoky",
}

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


def get_cluster_summary(df: pd.DataFrame, labels: np.ndarray, top_n: int = 8) -> dict:
    df = df.copy()
    df["cluster"] = labels
    summary = {}

    for cluster_id in sorted(df["cluster"].unique()):
        cluster_df = df[df["cluster"] == cluster_id]
        all_notes = []
        for _, row in cluster_df.iterrows():
            all_notes.extend(get_all_notes(row))

        top_notes = [note for note, _ in Counter(all_notes).most_common(top_n)]
        avg_rating = cluster_df["rating"].mean()

        summary[cluster_id] = {
            "label": CLUSTER_LABELS.get(cluster_id, f"Cluster {cluster_id}"),
            "top_notes": top_notes,
            "count": len(cluster_df),
            "avg_rating": round(avg_rating, 2),
            "perfumes": cluster_df["name"].tolist(),
        }

    return summary


def get_cluster_for_perfume(perfume_idx: int, labels: np.ndarray) -> dict:
    cluster_id = int(labels[perfume_idx])
    return {
        "cluster_id": cluster_id,
        "label": CLUSTER_LABELS.get(cluster_id, f"Cluster {cluster_id}"),
    }


def get_note_to_clusters(df: pd.DataFrame, labels: np.ndarray) -> dict[str, list[int]]:
    df = df.copy()
    df["cluster"] = labels
    note_clusters: dict[str, set] = {}

    for _, row in df.iterrows():
        for note in get_all_notes(row):
            note_clusters.setdefault(note, set()).add(int(row["cluster"]))

    return {note: sorted(clusters) for note, clusters in note_clusters.items()}