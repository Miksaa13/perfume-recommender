import sys
from pathlib import Path

import streamlit as st
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.recommender import (
    load_data,
    load_models,
    get_unique_notes,
    recommend_by_name,
    recommend_by_notes,
)
from models.clustering import get_cluster_summary, get_cluster_for_perfume, CLUSTER_LABELS

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Perfume Recommender",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { background-color: #0f0f0f; }
    .block-container { padding-top: 2rem; }

    .perfume-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #e8c99450;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 0.8rem;
        transition: border-color 0.2s;
    }
    .perfume-card:hover { border-color: #e8c994; }

    .perfume-name {
        font-size: 1.1rem;
        font-weight: 700;
        color: #e8c994;
        margin-bottom: 0.2rem;
    }
    .perfume-brand {
        font-size: 0.85rem;
        color: #a0a0b0;
        margin-bottom: 0.4rem;
    }
    .perfume-notes {
        font-size: 0.8rem;
        color: #c0c0d0;
        font-style: italic;
    }
    .badge {
        display: inline-block;
        padding: 0.15rem 0.5rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.3rem;
    }
    .badge-rating { background: #e8c99420; color: #e8c994; border: 1px solid #e8c99440; }
    .badge-genre  { background: #9b59b620; color: #c39bd3; border: 1px solid #9b59b640; }
    .badge-cluster { background: #2ecc7120; color: #82e0aa; border: 1px solid #2ecc7140; }
    .badge-score { background: #3498db20; color: #85c1e9; border: 1px solid #3498db40; }

    .cluster-box {
        background: linear-gradient(135deg, #1a2a1a 0%, #0d1a0d 100%);
        border: 1px solid #2ecc7150;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.5rem;
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #e8c994, #c39bd3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .hero-sub {
        color: #808090;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .section-header {
        color: #e8c994;
        font-size: 1.1rem;
        font-weight: 600;
        border-bottom: 1px solid #e8c99430;
        padding-bottom: 0.3rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Učitavam modele...")
def get_resources():
    df = load_data()
    sim_matrix, tfidf_mat, tfidf_vec, kmeans = load_models()
    unique_notes = get_unique_notes(df)
    cluster_summary = get_cluster_summary(df, kmeans.labels_)
    return df, sim_matrix, tfidf_mat, tfidf_vec, kmeans, unique_notes, cluster_summary


df, sim_matrix, tfidf_mat, tfidf_vec, kmeans, unique_notes, cluster_summary = get_resources()



def render_perfume_card(row: pd.Series, show_score: bool = True):
    cluster_label = CLUSTER_LABELS.get(int(row["cluster"]), f"Cluster {int(row['cluster'])}")
    score_html = ""
    if show_score and "similarity_score" in row.index:
        score_html = f'<span class="badge badge-score">⚡ {row["similarity_score"]:.2f}</span>'

    notes_preview = row["all_notes"][:80] + "..." if len(row.get("all_notes", "")) > 80 else row.get("all_notes", "")

    st.markdown(f"""
    <div class="perfume-card">
        <div class="perfume-name">{row["name"]}</div>
        <div class="perfume-brand">{row["brand"]}</div>
        <div style="margin: 0.4rem 0;">
            <span class="badge badge-rating">⭐ {row["rating"]}</span>
            <span class="badge badge-genre">{row["genre"]}</span>
            <span class="badge badge-cluster">{cluster_label}</span>
            {score_html}
        </div>
        <div class="perfume-notes">🌿 {notes_preview}</div>
    </div>
    """, unsafe_allow_html=True)


with st.sidebar:
    st.markdown("### Perfume Recommender")
    st.markdown("---")

    mode = st.radio(
        "Modo pretrage:",
        ["🔍 Pretraga po imenu", "🎵 Izbor nota"],
        index=0,
    )

    st.markdown("---")
    top_n = st.slider("Broj preporuka", min_value=3, max_value=15, value=8)

    st.markdown("---")
    genre_filter = st.selectbox(
        "Filter po tipu:",
        ["All", "Men", "Women", "Unisex"],
    )

    st.markdown("---")
    with st.expander("📊 Klasteri"):
        for cid, info in cluster_summary.items():
            st.markdown(f"""
            <div class="cluster-box">
                <b style="color:#e8c994">{info['label']}</b><br>
                <small style="color:#a0a0b0">{info['count']} parfema · ⭐ {info['avg_rating']}</small><br>
                <small style="color:#808090">{', '.join(info['top_notes'][:5])}</small>
            </div>
            """, unsafe_allow_html=True)

st.markdown('<div class="hero-title">🌸 Perfume Recommender</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Pronađi savršen parfem na osnovu onoga što voliš.</div>', unsafe_allow_html=True)

if mode == "🔍 Pretraga po imenu":
    st.markdown('<div class="section-header">Pretraži po imenu parfema</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        perfume_input = st.text_input(
            "",
            placeholder="npr. Layton, Aventus, Bleu de Chanel...",
            label_visibility="collapsed",
        )
    with col2:
        same_cluster = st.checkbox("Samo isti klaster", value=False)

    if perfume_input:
        mask_exact = df["name"].str.lower() == perfume_input.strip().lower()
        mask_partial = df["name"].str.lower().str.contains(perfume_input.strip().lower(), na=False)

        if not mask_partial.any():
            st.warning(f"Nije pronađen parfem sličan **{perfume_input}**")
        else:

            matches = df[mask_partial]["name"].tolist()

            if len(matches) > 1 and not mask_exact.any():
                selected = st.selectbox("Pronađeno više parfema — odaberi:", matches)
                perfume_input = selected

            sel_idx = df[df["name"].str.lower() == perfume_input.strip().lower()].index
            if len(sel_idx) > 0:
                sel_row = df.loc[sel_idx[0]]
                cluster_info = get_cluster_for_perfume(sel_idx[0], kmeans.labels_)
                all_n = sel_row.get("top_notes", ""), sel_row.get("middle_notes", ""), sel_row.get("base_notes", "")

                st.markdown("---")
                st.markdown(f"**Odabrani parfem:** `{sel_row['name']}` · {sel_row['brand']}")
                st.markdown(f"Klaster: **{cluster_info['label']}**")
                st.markdown("---")

            results = recommend_by_name(
                perfume_input, df, sim_matrix, kmeans,
                top_n=top_n, same_cluster_only=same_cluster
            )

            if results.empty:
                st.info("Nema dovoljno rezultata za ovaj filter.")
            else:
                st.markdown(f"<div class='section-header'>Top {len(results)} preporuka</div>", unsafe_allow_html=True)
                cols = st.columns(2)
                for i, (_, row) in enumerate(results.iterrows()):
                    with cols[i % 2]:
                        render_perfume_card(row)


else:
    st.markdown('<div class="section-header">Odaberi note koje voliš</div>', unsafe_allow_html=True)

    # Multiselect sa svim notama
    selected = st.multiselect(
        "",
        options=unique_notes,
        placeholder="Počni da kucaš notu... (npr. Vanilla, Oud, Bergamot)",
        label_visibility="collapsed",
    )

    st.markdown("**Popularne note:**")
    quick_notes = [
        "Vanilla", "Oud", "Bergamot", "Sandalwood", "Rose",
        "Patchouli", "Amber", "Musk", "Jasmine", "Vetiver",
        "Cedar", "Iris", "Tobacco", "Leather", "Neroli"
    ]

    if "quick_selected" not in st.session_state:
        st.session_state.quick_selected = set()

    cols_quick = st.columns(5)
    for i, note in enumerate(quick_notes):
        with cols_quick[i % 5]:
            is_active = note in st.session_state.quick_selected
            if st.button(
                    f"{'✓ ' if is_active else ''}{note}",
                    key=f"quick_{note}",
                    use_container_width=True,
            ):
                if note in st.session_state.quick_selected:
                    st.session_state.quick_selected.discard(note)
                else:
                    st.session_state.quick_selected.add(note)
                st.rerun()

    all_selected = list(set(selected) | st.session_state.quick_selected)

    if all_selected:
        st.markdown(f"**Odabrano:** {', '.join(sorted(all_selected))}")
        st.markdown("---")

        results = recommend_by_notes(
            all_selected, df, tfidf_mat, tfidf_vec, kmeans,
            top_n=top_n,
            genre_filter=genre_filter if genre_filter != "All" else None,
        )

        if results.empty:
            st.info("Nema rezultata za odabrane note i filter.")
        else:
            st.markdown(f"<div class='section-header'>Top {len(results)} parfema za odabrane note</div>",
                        unsafe_allow_html=True)
            cols = st.columns(2)
            for i, (_, row) in enumerate(results.iterrows()):
                with cols[i % 2]:
                    render_perfume_card(row)
    else:
        st.info("Odaberi jednu ili više nota da dobiješ preporuke.")

        st.markdown("---")
        st.markdown('<div class="section-header">Mirisne porodice</div>', unsafe_allow_html=True)
        cols = st.columns(2)
        for i, (cid, info) in enumerate(cluster_summary.items()):
            with cols[i % 2]:
                st.markdown(f"""
                <div class="cluster-box">
                    <b style="color:#e8c994">{info['label']}</b>
                    <span style="color:#a0a0b0; font-size:0.85rem"> · {info['count']} parfema</span><br>
                    <small style="color:#9b59b6">⭐ {info['avg_rating']} prosečno</small><br>
                    <small style="color:#808090">{', '.join(info['top_notes'][:6])}</small>
                </div>
                """, unsafe_allow_html=True)