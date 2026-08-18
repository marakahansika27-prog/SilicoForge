def calculate_family_support(clusters, total_families=25):
    for c in clusters:
        unique_families = set(p['family_id'] for p in c['peaks'])
        c['support_count'] = len(unique_families)
        c['support_fraction'] = float(len(unique_families)) / total_families
        c['supporting_family_ids'] = list(unique_families)
    return clusters
