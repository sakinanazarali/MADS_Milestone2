import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def find_optimal_clusters_agglomerative(df, text_columns, numeric_columns, categorical_columns, n_clusters_range, max_features, n_components):
    """
    find optimal number of clusters using PCA and agglomerative clustering

    """
    # vectorize using tfidf
    vectorizer = TfidfVectorizer(max_features=max_features)
    text_features = vectorizer.fit_transform(df[text_columns].apply(lambda x: ' '.join(x), axis=1))

    # standardize numeric columns
    scaler = StandardScaler()
    numeric_features = scaler.fit_transform(df[numeric_columns])

    # one hot encoding categorical columns
    encoder = OneHotEncoder()
    categorical_features = encoder.fit_transform(df[categorical_columns])

    # combine all the features
    processed_features = pd.concat([pd.DataFrame(text_features.toarray()), pd.DataFrame(numeric_features), pd.DataFrame(categorical_features.toarray())], axis=1)

    # apply PCA for dimensionality reduction
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(processed_features)

    # find optimal clusters using silhoutette method
    sil_scores = []
    db_scores = []
    ch_scores = []
    for k in n_clusters_range:
        agglomerative = AgglomerativeClustering(n_clusters=k)
        labels = agglomerative.fit_predict(pca_features)
        sil_scores.append(silhouette_score(pca_features, labels))
        db_scores.append(davies_bouldin_score(pca_features, labels))
        ch_scores.append(calinski_harabasz_score(pca_features, labels))

    # list optimum num of clusters based on different methods
    optimal_n_clusters_silhouette = n_clusters_range[np.argmax(sil_scores)]
    optimal_n_clusters_davies_bouldin = n_clusters_range[np.argmin(db_scores)]
    optimal_n_clusters_calinski_harabasz = n_clusters_range[np.argmax(ch_scores)]

    return {
        "silhouette method": optimal_n_clusters_silhouette,
        "davies bouldin index": optimal_n_clusters_davies_bouldin,
        "calinski harabasz Index": optimal_n_clusters_calinski_harabasz,
    }