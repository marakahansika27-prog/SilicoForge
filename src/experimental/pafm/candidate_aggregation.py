import numpy as np

def cluster_candidates(all_peaks, support_radius=5.0):
    """
    Groups canonical candidates that represent approximately the same spatial location.
    all_peaks: list of dicts with 'canonical_x', 'canonical_y', 'score', 'family_id', 'offset_x', 'offset_y'
    """
    clusters = []
    # Sort peaks by score descending
    sorted_peaks = sorted(all_peaks, key=lambda p: p['score'], reverse=True)
    
    for p in sorted_peaks:
        assigned = False
        for cluster in clusters:
            cx, cy = cluster['cluster_centroid']
            if np.sqrt((p['canonical_x'] - cx)**2 + (p['canonical_y'] - cy)**2) <= support_radius:
                cluster['peaks'].append(p)
                assigned = True
                break
        
        if not assigned:
            clusters.append({
                'cluster_centroid': (p['canonical_x'], p['canonical_y']),
                'peaks': [p]
            })
            
    # Update cluster centroids and metrics
    for c in clusters:
        xs = [p['canonical_x'] for p in c['peaks']]
        ys = [p['canonical_y'] for p in c['peaks']]
        scores = [p['score'] for p in c['peaks']]
        
        c['cluster_centroid'] = (float(np.mean(xs)), float(np.mean(ys)))
        c['x_std'] = float(np.std(xs)) if len(xs) > 1 else 0.0
        c['y_std'] = float(np.std(ys)) if len(ys) > 1 else 0.0
        
        max_dist = 0.0
        for i in range(len(xs)):
            for j in range(i+1, len(xs)):
                dist = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                if dist > max_dist:
                    max_dist = dist
                    
        c['max_pairwise_distance'] = float(max_dist)
        c['cluster_radius'] = support_radius
        c['mean_score'] = float(np.mean(scores))
        c['max_score'] = float(np.max(scores))
        c['score_std'] = float(np.std(scores)) if len(scores) > 1 else 0.0
        c['best_peak'] = c['peaks'][0] # already sorted
        
    return clusters
