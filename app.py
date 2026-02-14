import io
import numpy as np
import pandas as pd
import streamlit as st

from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF


st.set_page_config(page_title="Spotify NMF Recommender Dashboard", layout="wide")


# -------------------------
# Data Loading
# -------------------------
@st.cache_data(show_spinner=False)
def load_user_song_matrix_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    """
    Your file is CSV-like (even if named .xls). We'll parse as CSV.
    Expected shape: users as rows, songs as columns, values = play counts.
    """
    # Try UTF-8 first; fallback if needed
    try:
        s = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        s = file_bytes.decode("latin-1")

    df = pd.read_csv(io.StringIO(s))
    # first column is user id like "user_1"
    if df.columns[0].lower().startswith("unnamed") or df.columns[0] in ["", "user", "users"]:
        df = df.rename(columns={df.columns[0]: "user_id"})
    else:
        df = df.rename(columns={df.columns[0]: "user_id"})

    # ensure numeric song columns
    song_cols = [c for c in df.columns if c != "user_id"]
    df[song_cols] = df[song_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)
    return df


@st.cache_data(show_spinner=False)
def load_from_repo_default(path: str) -> pd.DataFrame | None:
    try:
        with open(path, "rb") as f:
            return load_user_song_matrix_from_bytes(f.read())
    except FileNotFoundError:
        return None


# -------------------------
# Modeling (NMF)
# -------------------------
@st.cache_resource(show_spinner=False)
def train_nmf(play_matrix: csr_matrix, n_components: int, max_iter: int, random_state: int):
    """
    Uses NMF for implicit-ish feedback (play counts).
    mu solver + KL divergence typically works well for count data.
    """
    model = NMF(
        n_components=n_components,
        init="nndsvda",
        solver="mu",
        beta_loss="kullback-leibler",
        max_iter=max_iter,
        random_state=random_state,
        alpha_W=0.0,
        alpha_H=0.0,
        l1_ratio=0.0,
    )
    W = model.fit_transform(play_matrix)
    H = model.components_
    return model, W, H


def recommend_for_user(user_index: int, play_matrix: csr_matrix, W: np.ndarray, H: np.ndarray, song_names: list[str], top_n: int):
    """
    Reconstruct scores for a user and recommend songs not already listened to.
    """
    user_scores = W[user_index] @ H  # (n_songs,)
    user_row = play_matrix.getrow(user_index).toarray().ravel()
    already_listened = user_row > 0

    user_scores = user_scores.copy()
    user_scores[already_listened] = -np.inf

    top_idx = np.argpartition(-user_scores, range(min(top_n, len(user_scores))))[:top_n]
    top_idx = top_idx[np.argsort(-user_scores[top_idx])]

    recs = pd.DataFrame({
        "song": [song_names[i] for i in top_idx],
        "score": user_scores[top_idx]
    })
    return recs


def top_popular_songs(df: pd.DataFrame, top_n: int):
    song_cols = [c for c in df.columns if c != "user_id"]
    totals = df[song_cols].sum(axis=0).sort_values(ascending=False).head(top_n)
    return pd.DataFrame({"song": totals.index, "total_plays": totals.values})


# -------------------------
# UI
# -------------------------
st.title("🎧 Spotify Recommendation System Dashboard (NMF)")

with st.sidebar:
    st.header("Data")
    st.caption("Load from repo file OR upload from here.")
    uploaded = st.file_uploader("Upload your user-song matrix file", type=["csv", "xls", "xlsx", "txt"])

    st.header("Model Controls")
    n_components = st.slider("Latent Factors (NMF components)", 10, 150, 50, 5)
    max_iter = st.slider("Max Iterations", 100, 1000, 300, 50)
    top_n = st.slider("Recommendations (Top N)", 5, 50, 15, 1)

    st.header("Dashboard Controls")
    popular_n = st.slider("Popular Songs Count", 10, 100, 20, 5)
    random_state = st.number_input("Random Seed", min_value=0, value=42, step=1)


# Load data
df = None
if uploaded is not None:
    df = load_user_song_matrix_from_bytes(uploaded.getvalue())
else:
    # default repo filename
    df = load_from_repo_default("spotify (1).xls")

if df is None:
    st.warning("No dataset found. Either upload your file in the sidebar OR add `spotify (1).xls` to the repo root.")
    st.stop()

# Prepare matrix
user_ids = df["user_id"].astype(str).tolist()
song_cols = [c for c in df.columns if c != "user_id"]
song_names = song_cols

X = csr_matrix(df[song_cols].values)

# Train model
with st.spinner("Training NMF model..."):
    model, W, H = train_nmf(X, n_components=n_components, max_iter=max_iter, random_state=int(random_state))

# -------------------------
# Layout
# -------------------------
col1, col2, col3 = st.columns([1.2, 1.2, 1.6])

with col1:
    st.subheader("Dataset Overview")
    st.metric("Users", df.shape[0])
    st.metric("Songs", len(song_cols))
    st.metric("Non-zero Plays", int(X.nnz))

    # sparsity
    total_cells = df.shape[0] * len(song_cols)
    sparsity = 1 - (X.nnz / total_cells)
    st.metric("Sparsity", f"{sparsity*100:.2f}%")

with col2:
    st.subheader("Top Popular Songs")
    pop = top_popular_songs(df, top_n=popular_n)
    st.dataframe(pop, use_container_width=True, height=360)

with col3:
    st.subheader("User Recommendation Panel")

    selected_user = st.selectbox("Select a user", user_ids, index=0)
    user_index = user_ids.index(selected_user)

    # user listening summary
    user_row = X.getrow(user_index).toarray().ravel()
    listened_count = int((user_row > 0).sum())
    total_user_plays = float(user_row.sum())
    st.write(f"**Listened songs:** {listened_count}  |  **Total plays:** {total_user_plays:.0f}")

    # top listened for this user
    top_listened_idx = np.argsort(-user_row)[:10]
    top_listened = pd.DataFrame({
        "song": [song_names[i] for i in top_listened_idx],
        "plays": user_row[top_listened_idx]
    })
    top_listened = top_listened[top_listened["plays"] > 0]

    cA, cB = st.columns(2)
    with cA:
        st.write("**Top listened (this user)**")
        st.dataframe(top_listened, use_container_width=True, height=260)

    with cB:
        st.write("**Recommended for this user (NMF)**")
        recs = recommend_for_user(user_index, X, W, H, song_names, top_n=top_n)
        st.dataframe(recs, use_container_width=True, height=260)

st.divider()

# Extra analytics
st.subheader("Extra Analytics")

c1, c2 = st.columns(2)
with c1:
    st.write("**Distribution: #songs listened per user**")
    listened_per_user = (df[song_cols].values > 0).sum(axis=1)
    st.bar_chart(pd.Series(listened_per_user).value_counts().sort_index())

with c2:
    st.write("**Distribution: total plays per user**")
    plays_per_user = df[song_cols].sum(axis=1)
    # bucketize for cleaner view
    bins = pd.cut(plays_per_user, bins=20)
    st.bar_chart(bins.value_counts().sort_index())

st.caption("Model: Non-negative Matrix Factorization (NMF) on play-count matrix. Recommendations exclude already-listened songs.")
