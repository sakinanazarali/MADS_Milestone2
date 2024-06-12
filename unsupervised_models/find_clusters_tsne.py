from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from yellowbrick.cluster import KElbowVisualizer
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import hstack

def find_optimal_clusters(df, text_columns, numeric_columns, categorical_columns, n_clusters_range, max_features, n_components):
    """
    find optimal number of clusters using t-sne
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
    processed_features = hstack([text_features, numeric_features, categorical_features])

    # set perplexity based on the number of samples
    # n_samples = processed_features.shape[0]
    # perplexity = min(30, n_samples - 1)

    # apply t-sne for dimensionality reduction
    # tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    tsne = TSNE(n_components=n_components, random_state=42)
    tsne_features = tsne.fit_transform(processed_features.toarray())

    # find optimal clusters using silhoutette method
    sil_scores = []
    for k in n_clusters_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(tsne_features)
        labels = kmeans.labels_
        sil_scores.append(silhouette_score(tsne_features, labels))

    # find optimal clusters using gap method
    visualizer = KElbowVisualizer(KMeans(), k=n_clusters_range, metric='distortion', timings=False)
    visualizer.fit(tsne_features)

    # find optimal clusters using davies-bouldin index
    db_scores = []
    for k in n_clusters_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(tsne_features)
        labels = kmeans.labels_
        db_scores.append(davies_bouldin_score(tsne_features, labels))

    # find optimal clusters using calinski-harabasz index
    ch_scores = []
    for k in n_clusters_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(tsne_features)
        labels = kmeans.labels_
        ch_scores.append(calinski_harabasz_score(tsne_features, labels))

    # find optimal clusters using avg within sum of square method (elbow method) (AWSS)
    awss_values = []
    for k in n_clusters_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(tsne_features)
        awss_values.append(kmeans.inertia_)

    # plot elbow method
    plt.figure(figsize=(10, 6))
    plt.plot(n_clusters_range, awss_values, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('AWSS')
    plt.title('Elbow Method with AWSS')
    plt.xticks(n_clusters_range)
    plt.grid()
    plt.show()

    # list optimum num of clusters based on different methods
    optimal_n_clusters_silhouette = n_clusters_range[np.argmax(sil_scores)]
    optimal_n_clusters_gap = visualizer.elbow_value_
    optimal_n_clusters_davies_bouldin = n_clusters_range[np.argmin(db_scores)]
    optimal_n_clusters_calinski_harabasz = n_clusters_range[np.argmax(ch_scores)]
    optimal_n_clusters_awss = n_clusters_range[np.argmin(awss_values)]

    return {
        "silhouette method": optimal_n_clusters_silhouette,
        "gap stat": optimal_n_clusters_gap,
        "davies bouldin index": optimal_n_clusters_davies_bouldin,
        "calinski harabasz Index": optimal_n_clusters_calinski_harabasz,
        "AWSS elbow method": optimal_n_clusters_awss
    }
