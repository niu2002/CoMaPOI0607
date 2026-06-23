# -*- coding: utf-8 -*-
"""
Offline script to generate Hierarchical Semantic IDs (HSID) for POIs.
Creates dataset_all/<dataset>/poi_hsid.json.
Supports a fallback rule-based clustering if sklearn is not installed.
"""
import argparse
import os
import json
import hashlib
import pandas as pd
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Build HSID for POIs")
    parser.add_argument("--dataset", type=str, default="ca", choices=["nyc", "tky", "ca"])
    parser.add_argument("--coarse_cell", type=float, default=0.05, help="Coarse grid size (degrees)")
    parser.add_argument("--fine_cell", type=float, default=0.01, help="Fine grid size (degrees)")
    parser.add_argument("--cluster_k", type=int, default=256, help="Number of semantic clusters (K)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Resolve paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    poi_info_path = os.path.join(project_root, "dataset_all", args.dataset, f"{args.dataset}_poi_info.csv")
    out_dir = os.path.join(project_root, "dataset_all", args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "poi_hsid.json")
    
    if not os.path.exists(poi_info_path):
        # Try parent dir just in case
        poi_info_path_alt = os.path.join(project_root, f"{args.dataset}_poi_info.csv")
        if os.path.exists(poi_info_path_alt):
            poi_info_path = poi_info_path_alt
        else:
            raise FileNotFoundError(f"POI info file not found: {poi_info_path}")
        
    df = pd.read_csv(poi_info_path)
    print(f"Loaded {len(df)} POIs for dataset '{args.dataset}'.")
    
    # Calculate geographical bounds and baselines
    min_lat, min_lon = df['lat'].min(), df['lon'].min()
    
    # Level 1 & 2: Grid region IDs
    coarse_regions = []
    fine_regions = []
    
    for _, row in df.iterrows():
        lat, lon = row['lat'], row['lon']
        cx = int((lat - min_lat) / args.coarse_cell)
        cy = int((lon - min_lon) / args.coarse_cell)
        c_reg = f"R_{cx}_{cy}"
        
        fx = int((lat - min_lat) / args.fine_cell)
        fy = int((lon - min_lon) / args.fine_cell)
        f_reg = f"{c_reg}_{fx}_{fy}"
        
        coarse_regions.append(c_reg)
        fine_regions.append(f_reg)
        
    df['coarse_region'] = coarse_regions
    df['fine_region'] = fine_regions
    
    # Level 4: Semantic Cluster (KMeans coords + category, fallback to hash-based if sklearn missing)
    use_kmeans = False
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import OneHotEncoder
        use_kmeans = True
    except ImportError:
        print("[WARN] sklearn is not installed. Using deterministic hash-based clustering fallback.")
        
    if use_kmeans:
        try:
            max_lat, max_lon = df['lat'].max(), df['lon'].max()
            df['lat_norm'] = (df['lat'] - min_lat) / (max_lat - min_lat + 1e-9)
            df['lon_norm'] = (df['lon'] - min_lon) / (max_lon - min_lon + 1e-9)
            
            encoder = OneHotEncoder(sparse_output=False)
            cat_encoded = encoder.fit_transform(df[['category']])
            
            features = np.hstack([df[['lat_norm', 'lon_norm']].values, cat_encoded])
            
            print(f"Running KMeans cluster (K={args.cluster_k})...")
            kmeans = KMeans(n_clusters=args.cluster_k, random_state=args.seed, n_init=10)
            df['cluster_label'] = kmeans.fit_predict(features)
        except Exception as e:
            print(f"[WARN] KMeans clustering failed ({e}). Falling back to hash-based clustering.")
            use_kmeans = False
            
    if not use_kmeans:
        # Fallback: Deterministic MD5 Hash partition into cluster_k bins
        cluster_labels = []
        for _, row in df.iterrows():
            combined_key = f"{row['category']}_{row['fine_region']}"
            hash_val = int(hashlib.md5(combined_key.encode('utf-8')).hexdigest(), 16)
            cluster_labels.append(hash_val % args.cluster_k)
        df['cluster_label'] = cluster_labels
        
    # Assemble HSID mapping
    hsid_dict = {}
    for _, row in df.iterrows():
        poi_id_str = str(int(row['poi_id']))
        coarse = row['coarse_region']
        fine = row['fine_region']
        category = str(row['category']).strip()
        cluster = f"SC_{row['cluster_label']:03d}"
        
        hsid_text = f"coarse_region={coarse}; fine_region={fine}; category={category}; semantic_cluster={cluster}"
        
        hsid_dict[poi_id_str] = {
            "coarse_region": coarse,
            "fine_region": fine,
            "category": category,
            "semantic_cluster": cluster,
            "hsid_text": hsid_text
        }
        
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(hsid_dict, f, indent=2, ensure_ascii=False)
        
    print(f"HSID data successfully saved to: {out_file}")

if __name__ == "__main__":
    main()
