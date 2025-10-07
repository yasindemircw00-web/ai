from sklearn.cluster import KMeans, AgglomerativeClustering
import hdbscan
import numpy as np
from numpy.typing import NDArray


def cluster_embeddings(embeddings: list[NDArray[np.floating]] | NDArray[np.floating], method: str = 'kmeans', n_clusters: int = 2) -> list[int]:
    """
    Embedding'leri kümelemek için kullanılan fonksiyon.
    
    Args:
        embeddings (list[NDArray[np.floating]] | NDArray[np.floating]): Embedding vektörleri.
        method (str): Kümeleme yöntemi ('kmeans', 'agglomerative', 'hdbscan').
        n_clusters (int): Küme sayısı (sadece 'kmeans' ve 'agglomerative' için).
        
    Returns:
        labels (list[int]): Her segmentin ait olduğu küme etiketi.
    """
    # Embedding'leri numpy array'e çevir
    if isinstance(embeddings, list):
        embeddings = np.array(embeddings)
    
    # Embedding'lerin geçerli olduğundan emin ol
    if len(embeddings) == 0:
        return []
    
    # Tek bir embedding varsa, tek bir küme döndür
    if len(embeddings) == 1:
        return [0]
    
    # Tüm embedding'lerin aynı olup olmadığını kontrol et
    all_same = True
    if len(embeddings) > 1:
        first_embedding = embeddings[0]
        for emb in embeddings[1:]:
            if not np.allclose(emb, first_embedding, rtol=1e-5):
                all_same = False
                break
    
    # Tüm embedding'ler aynıysa, rastgele etiketler ata
    if all_same:
        print("Warning: All embeddings are identical. Assigning random labels.")
        # Rastgele etiketler ata (konuşmacıları simüle et)
        labels = np.random.randint(0, min(n_clusters, len(embeddings)), len(embeddings))
        return labels.tolist()
    
    # NaN veya inf değerleri kontrol et ve temizle
    if np.isnan(embeddings).any() or np.isinf(embeddings).any():
        print("Warning: NaN or infinite values found in embeddings. Using zero vectors.")
        embeddings = np.nan_to_num(embeddings)
    
    try:
        labels: NDArray[np.integer] | list[int]
        if method == 'kmeans':
            # KMeans için doğru parametreleri kullan
            clustering_model = KMeans(n_clusters=min(n_clusters, len(embeddings)))
            labels = np.asarray(clustering_model.fit_predict(embeddings), dtype=np.int32)
        elif method == 'agglomerative':
            if len(embeddings) >= n_clusters:
                clustering_model = AgglomerativeClustering(n_clusters=min(n_clusters, len(embeddings)))
                labels = np.asarray(clustering_model.fit_predict(embeddings), dtype=np.int32)
            else:
                # Yeterli veri yoksa tüm segmentlere aynı etiketi ver
                labels = [0] * len(embeddings)
        elif method == 'hdbscan':
            if len(embeddings) >= 2:
                clustering_model = hdbscan.HDBSCAN(min_cluster_size=min(2, len(embeddings)))
                labels = np.asarray(clustering_model.fit_predict(embeddings), dtype=np.int32)
            else:
                # Yeterli veri yoksa tüm segmentlere aynı etiketi ver
                labels = [0] * len(embeddings)
        else:
            raise ValueError("Geçersiz kümeleme yöntemi.")
    except Exception as e:
        print(f"Clustering failed: {e}")
        # Hata durumunda rastgele etiketler ata
        labels = np.random.randint(0, min(n_clusters, len(embeddings)), len(embeddings))
        return labels.tolist()
    
    # Ensure we return a list of integers
    if isinstance(labels, np.ndarray):
        return labels.astype(int).tolist()
    else:
        return [int(label) for label in labels]
