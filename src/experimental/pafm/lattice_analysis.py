import numpy as np

def lattice_aware_nms(clusters, nms_radius=50.0, top_k=20):
    """
    Retains spatially distinct candidates. 
    Lattice detection marks structurally distinct candidates, but NMS ensures we don't have overlapping clusters.
    """
    # Sort clusters by max_score
    sorted_clusters = sorted(clusters, key=lambda c: c['max_score'], reverse=True)
    
    distinct_clusters = []
    for c in sorted_clusters:
        is_distinct = True
        cx, cy = c['cluster_centroid']
        for dc in distinct_clusters:
            dcx, dcy = dc['cluster_centroid']
            if np.sqrt((cx - dcx)**2 + (cy - dcy)**2) <= nms_radius:
                is_distinct = False
                break
        if is_distinct:
            distinct_clusters.append(c)
        if len(distinct_clusters) >= top_k:
            break
            
    # Analyze lattice relationships among distinct clusters
    lattice_pairs = []
    for i in range(len(distinct_clusters)):
        for j in range(i+1, len(distinct_clusters)):
            dx = distinct_clusters[i]['cluster_centroid'][0] - distinct_clusters[j]['cluster_centroid'][0]
            dy = distinct_clusters[i]['cluster_centroid'][1] - distinct_clusters[j]['cluster_centroid'][1]
            lattice_pairs.append({
                'idx_i': i, 'idx_j': j, 
                'dx': float(dx), 'dy': float(dy), 'dist': float(np.sqrt(dx**2 + dy**2))
            })
            
    return distinct_clusters, lattice_pairs
