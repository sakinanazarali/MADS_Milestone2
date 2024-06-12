import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

def unsupervised_pca_agglomerative(df, text_columns, numeric_columns, categorical_columns, target_column, n_clusters, max_features, n_components):
    """
    standardize names in the target col using pca and agglomerative clustering
    """
    # vectorize using tfidf
    vectorizer = TfidfVectorizer(max_features=max_features)
    text_features = vectorizer.fit_transform(df[text_columns].apply(lambda x: ' '.join(x), axis=1))

    # standardize numeric columns -- may not be necessary
    scaler = StandardScaler()
    numeric_features = scaler.fit_transform(df[numeric_columns])

    # apply one hot encoding for categorical columns
    encoder = OneHotEncoder()
    categorical_features = encoder.fit_transform(df[categorical_columns])

    # combine all features
    processed_features = pd.concat([pd.DataFrame(text_features.toarray()), pd.DataFrame(numeric_features), pd.DataFrame(categorical_features.toarray())], axis=1)

    # apply PCA for dimensionality reduction
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(processed_features)

    # use agglomerative clustering
    agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
    clusters = agglomerative.fit_predict(pca_features)

    # assign the cluster labels to the data
    df['cluster'] = clusters

    # find the most common name in each cluster and add it to the standardized_names dict
    standardized_names = {}
    for cluster_id in range(n_clusters):
        cluster_data = df[df['cluster'] == cluster_id][target_column]
        most_common_name = cluster_data.mode()[0]
        standardized_names[cluster_id] = most_common_name

    # standardize name for each of the entities using the most common name detected
    df['standardized_name'] = df['cluster'].map(standardized_names)

    return df
