import numpy as np

def classify_ambiguity(clusters):
    if len(clusters) < 2:
        return "CLEAR", 1.0, 1.0
        
    peak1 = clusters[0]['max_score']
    peak2 = clusters[1]['max_score']
    
    margin = float(peak1 - peak2)
    ratio = float(peak1 / peak2 if peak2 != 0 else 999.0)
    
    if peak1 < 0.3:
        classification = "LOW-EVIDENCE"
    elif margin < 0.05:
        # Check if periodic
        dist = np.sqrt((clusters[0]['cluster_centroid'][0] - clusters[1]['cluster_centroid'][0])**2 + 
                       (clusters[0]['cluster_centroid'][1] - clusters[1]['cluster_centroid'][1])**2)
        if dist > 25.0: # Arbitrary threshold for periodicity vs just noisy local max
            classification = "PERIODICALLY AMBIGUOUS"
        else:
            classification = "AMBIGUOUS"
    else:
        classification = "CLEAR"
        
    return classification, margin, ratio
