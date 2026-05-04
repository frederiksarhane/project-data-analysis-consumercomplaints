import re
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans

sns.set(style="whitegrid")

CSV_PATH = r"C:\Users\frede\DataAnalysis\complaints\complaints.csv"  # !!!!!

def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    return df

def filter_and_sample(df: pd.DataFrame, max_rows: Optional[int] = 30000) -> pd.DataFrame:
    text_col = "Consumer complaint narrative"
    if text_col not in df.columns:
        raise KeyError(f"Spalte '{text_col}' nicht gefunden. Vorhandene Spalten: {df.columns.tolist()}")
    df = df[df[text_col].notna()].copy()
    df = df[df[text_col].astype(str).str.strip() != ""]
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42)
    df.reset_index(drop=True, inplace=True)
    return df

def basic_clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def add_clean_text_column(df: pd.DataFrame) -> pd.DataFrame:
    text_col = "Consumer complaint narrative"
    if text_col not in df.columns:
        raise KeyError(f"Spalte '{text_col}' nicht gefunden.")
    df = df.copy()
    df["clean_text"] = df[text_col].apply(basic_clean_text)
    return df

def vectorize_texts_count(texts, max_features: int = 5000):
    vectorizer = CountVectorizer(max_features=max_features, stop_words="english")
    X_counts = vectorizer.fit_transform(texts)
    return X_counts, vectorizer

def vectorize_texts_tfidf(texts, max_features: int = 5000):
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    X_tfidf = vectorizer.fit_transform(texts)
    return X_tfidf, vectorizer

def fit_lda(X_counts, n_topics: int = 8, max_iter: int = 10, random_state: int = 42):
    lda = LatentDirichletAllocation(
        n_components=n_topics, max_iter=max_iter,
        learning_method="batch", random_state=random_state
    )
    lda.fit(X_counts)
    return lda

def print_top_words(model, feature_names, n_top_words: int = 10):
    for topic_idx, topic in enumerate(model.components_):
        top_indices = topic.argsort()[:-n_top_words - 1:-1]
        top_terms = [feature_names[i] for i in top_indices]
        print(f"Topic #{topic_idx}: {' | '.join(top_terms)}")

def fit_kmeans(X_tfidf, n_clusters: int = 8, random_state: int = 42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans.fit(X_tfidf)
    return kmeans

def print_top_terms_per_cluster(kmeans, feature_names, n_top_terms: int = 10):
    centers = kmeans.cluster_centers_
    for cluster_idx, center in enumerate(centers):
        top_indices = center.argsort()[:-n_top_terms - 1:-1]
        top_terms = [feature_names[i] for i in top_indices]
        print(f"Cluster #{cluster_idx}: {' | '.join(top_terms)}")

def plot_lda_topic_top_words(lda_model, feature_names, topic_idx: int,
                             n_top_words: int = 10,
                             filename: str = "lda_topic_topwords.png"):
    topic = lda_model.components_[topic_idx]
    top_indices = topic.argsort()[:-n_top_words - 1:-1]
    top_terms = [feature_names[i] for i in top_indices]
    top_scores = topic[top_indices]

    plt.figure(figsize=(8, 4))
    sns.barplot(x=top_scores, y=top_terms, orient="h", color="steelblue")
    plt.xlabel("Wichtungswert im Thema")
    plt.ylabel("Wort")
    plt.title(f"Top-{n_top_words} Wörter für LDA-Topic #{topic_idx}")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Grafik '{filename}' gespeichert.")

def plot_cluster_size_distribution(df: pd.DataFrame,
                                   cluster_col: str = "cluster_id",
                                   filename: str = "cluster_sizes.png"):
    counts = df[cluster_col].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    sns.barplot(x=counts.index.astype(str), y=counts.values, color="steelblue")
    plt.xlabel("Cluster-ID")
    plt.ylabel("Anzahl Beschwerden")
    plt.title("Verteilung der Clustergrößen")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Grafik '{filename}' gespeichert.")

def plot_top_products(df: pd.DataFrame, top_n: int = 10,
                      filename: str = "top_products.png"):
    if "Product" not in df.columns:
        print("Spalte 'Product' nicht vorhanden, überspringe Produktgrafik.")
        return
    counts = df["Product"].value_counts().head(top_n)
    plt.figure(figsize=(8, 4))
    sns.barplot(x=counts.values, y=counts.index, orient="h", color="steelblue")
    plt.xlabel("Anzahl Beschwerden")
    plt.ylabel("Produkt")
    plt.title(f"Top-{top_n} Finanzprodukte im Beschwerdedatensatz")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Grafik '{filename}' gespeichert.")

def main():
    df_raw = load_raw_data(CSV_PATH)
    print("Rohdaten geladen.")
    print("Anzahl Zeilen (roh):", len(df_raw))
    print("Spaltennamen (Auszug):", df_raw.columns.tolist()[:10], "...")

    df_subset = filter_and_sample(df_raw, max_rows=30000)
    print("Anzahl Zeilen nach Filter:", len(df_subset))

    wichtige_spalten = ["Date received", "Product", "Issue",
                        "Consumer complaint narrative", "Company"]
    vorhandene_spalten = [c for c in wichtige_spalten if c in df_subset.columns]
    print("\nBeispielzeilen (Kopf):")
    print(df_subset[vorhandene_spalten].head(5).to_string())

    df_clean = add_clean_text_column(df_subset)
    print("\nBeispiel Original vs. Clean-Text:")
    for i in range(3):
        print("-" * 70)
        print("ORIGINAL:", df_clean.loc[i, "Consumer complaint narrative"][:200])
        print("CLEAN:   ", df_clean.loc[i, "clean_text"][:200])

    df_clean.to_csv("complaints_clean_subset.csv", index=False)
    print("\n'complaints_clean_subset.csv' gespeichert.")

    texts = df_clean["clean_text"].tolist()

    print("\nCountVectorizer...")
    X_counts, count_vect = vectorize_texts_count(texts, max_features=5000)
    print("Shape Count-Matrix:", X_counts.shape)

    print("\nTfidfVectorizer...")
    X_tfidf, tfidf_vect = vectorize_texts_tfidf(texts, max_features=5000)
    print("Shape TF-IDF-Matrix:", X_tfidf.shape)

    print("\nTrainiere LDA-Modell (8 Topics)...")
    lda_model = fit_lda(X_counts, n_topics=8, max_iter=10, random_state=42)

    feature_names_count = count_vect.get_feature_names_out()
    print("\nTop-Wörter pro Thema (LDA):")
    print_top_words(lda_model, feature_names_count, n_top_words=10)

    doc_topic_dist = lda_model.transform(X_counts)
    df_clean["lda_topic_main"] = doc_topic_dist.argmax(axis=1)

    plot_lda_topic_top_words(lda_model, feature_names_count,
                             topic_idx=0, n_top_words=10,
                             filename="lda_topic0_topwords.png")

    print("\nK-Means Clustering (8 Cluster)...")
    kmeans_model = fit_kmeans(X_tfidf, n_clusters=8, random_state=42)

    feature_names_tfidf = tfidf_vect.get_feature_names_out()
    print("\nTop-Terme pro Cluster (K-Means):")
    print_top_terms_per_cluster(kmeans_model, feature_names_tfidf, n_top_terms=10)

    df_clean["cluster_id"] = kmeans_model.labels_

    print("\nClustergrößen:")
    print(df_clean["cluster_id"].value_counts().sort_index())

    plot_cluster_size_distribution(df_clean, cluster_col="cluster_id",
                                   filename="cluster_sizes.png")

    plot_top_products(df_clean, top_n=10, filename="top_products.png")

    print("\n--- Beispielbeschwerden pro Cluster ---")
    for cid in range(8):
        print(f"\n{'='*70}")
        print(f"CLUSTER {cid}")
        beispiele = df_clean[df_clean["cluster_id"] == cid][
            "Consumer complaint narrative"].head(2)
        for j, text in enumerate(beispiele, start=1):
            print(f"  Beispiel {j}: {str(text)[:300]}")

    df_clean.to_csv("complaints_clean_with_topics.csv", index=False)
    print("\n'complaints_clean_with_topics.csv' gespeichert.")
    df_raw.to_csv("complaints_raw_backup.csv", index=False)
    print("'complaints_raw_backup.csv' gespeichert.")
    print("\nFertig! Erzeugte Grafiken: lda_topic0_topwords.png | cluster_sizes.png | top_products.png")

if __name__ == "__main__":
    main()
