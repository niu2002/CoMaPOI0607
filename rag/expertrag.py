import os
import json
import re
import numpy as np
import pandas as pd
from pathlib import Path

# Haversine distance helper (vectorized)
def haversine_distance(lat1, lon1, lats, lons):
    r = 6371.0  # Earth's radius in km
    phi1 = np.radians(lat1)
    phi2 = np.radians(lats)
    delta_phi = np.radians(lats - lat1)
    delta_lambda = np.radians(lons - lon1)
    a = np.sin(delta_phi/2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2.0)**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0-a))
    return r * c

class ExpertRAG:
    def __init__(self, rag_finder):
        self.rag_finder = rag_finder
        self.poi_data = rag_finder.poi_data
        self.max_id = rag_finder.max_id
        
        # Load and parse coordinates
        self.poi_lats = self.poi_data['lat'].values
        self.poi_lons = self.poi_data['lon'].values
        self.poi_categories = self.poi_data['category'].values
        self.poi_ids = self.poi_data['poi_id'].values
        self.poi_id_to_idx = {pid: idx for idx, pid in enumerate(self.poi_ids)}
        
        # Load HSID data directly if not provided by rag_finder
        self.hsid_data = getattr(self.rag_finder, "hsid_data", {})
        if not self.hsid_data:
            project_root = Path(__file__).resolve().parent.parent
            hsid_path = project_root / "dataset_all" / self.rag_finder.data / "poi_hsid.json"
            if hsid_path.exists():
                print(f"[ExpertRAG] Active loading HSID mapping from {hsid_path}")
                with open(hsid_path, "r", encoding="utf-8") as f:
                    self.hsid_data = json.load(f)
            else:
                print(f"[ExpertRAG] [WARN] HSID file not found at {hsid_path}")

        # Initialize transition and temporal stats from training set
        self.poi_transitions = {}
        self.cat_transitions = {}
        self.coarse_transitions = {}
        self.fine_transitions = {}
        self.cluster_transitions = {}
        self.hourly_cat_probs = {}
        self._build_historical_stats()

    def _build_historical_stats(self):
        print("[ExpertRAG] Building historical stats from training dataset...")
        project_root = Path(__file__).resolve().parent.parent
        train_file = project_root / "dataset_all" / f"{self.rag_finder.data}_train.jsonl"
        
        if not train_file.exists():
            print(f"[ExpertRAG] [WARN] Training file not found at {train_file}, skipping stats building.")
            return

        # Initialize hourly counts
        # hourly_cat_counts[hour][category] = count
        hourly_cat_counts = {h: {} for h in range(24)}

        with open(train_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    msg = json.loads(line)
                    user_content = msg["messages"][1]["content"]
                    assistant_content = json.loads(msg["messages"][2]["content"])
                    label_poi = int(assistant_content.get("next_poi_id", -1))
                    
                    # Parse trajectory to extract transitions and hours
                    matches = re.findall(r'At (\d{4}-\d{2}-\d{2}) (\d{2}):\d{2}:\d{2}.*?visit POI ID (\d+) \(([^,]*),', user_content)
                    if not matches:
                        continue
                    
                    # 1. Temporal: Associate labels with target hour
                    last_hour = int(matches[-1][1])
                    if label_poi in self.poi_id_to_idx:
                        label_idx = self.poi_id_to_idx[label_poi]
                        label_cat = self.poi_categories[label_idx]
                        hourly_cat_counts[last_hour][label_cat] = hourly_cat_counts[last_hour].get(label_cat, 0) + 1

                    # 2. Transitions: Record POI, Category, and HSID-level transitions
                    for i in range(len(matches) - 1):
                        p1 = int(matches[i][2])
                        p2 = int(matches[i+1][2])
                        c1 = matches[i][3].strip()
                        c2 = matches[i+1][3].strip()
                        
                        if p1 not in self.poi_transitions:
                            self.poi_transitions[p1] = {}
                        self.poi_transitions[p1][p2] = self.poi_transitions[p1].get(p2, 0) + 1

                        if c1 not in self.cat_transitions:
                            self.cat_transitions[c1] = {}
                        self.cat_transitions[c1][c2] = self.cat_transitions[c1].get(c2, 0) + 1

                        # HSID multi-scale spatial transitions
                        h1 = self.hsid_data.get(str(p1))
                        h2 = self.hsid_data.get(str(p2))
                        if h1 and h2:
                            coarse1 = h1.get("coarse_region")
                            coarse2 = h2.get("coarse_region")
                            if coarse1 and coarse2:
                                if coarse1 not in self.coarse_transitions:
                                    self.coarse_transitions[coarse1] = {}
                                self.coarse_transitions[coarse1][coarse2] = self.coarse_transitions[coarse1].get(coarse2, 0) + 1

                            fine1 = h1.get("fine_region")
                            fine2 = h2.get("fine_region")
                            if fine1 and fine2:
                                if fine1 not in self.fine_transitions:
                                    self.fine_transitions[fine1] = {}
                                self.fine_transitions[fine1][fine2] = self.fine_transitions[fine1].get(fine2, 0) + 1

                            cluster1 = h1.get("semantic_cluster")
                            cluster2 = h2.get("semantic_cluster")
                            if cluster1 and cluster2:
                                if cluster1 not in self.cluster_transitions:
                                    self.cluster_transitions[cluster1] = {}
                                self.cluster_transitions[cluster1][cluster2] = self.cluster_transitions[cluster1].get(cluster2, 0) + 1

                except Exception as e:
                    continue

        # Normalize hourly category counts to probabilities
        for h in range(24):
            total = sum(hourly_cat_counts[h].values())
            if total > 0:
                self.hourly_cat_probs[h] = {cat: count / total for cat, count in hourly_cat_counts[h].items()}
            else:
                self.hourly_cat_probs[h] = {}

        print(f"[ExpertRAG] Transition stats built.")
        print(f"  Unique POI transition sources: {len(self.poi_transitions)}")
        print(f"  Unique Coarse transition sources: {len(self.coarse_transitions)}")
        print(f"  Unique Fine transition sources: {len(self.fine_transitions)}")
        print(f"  Unique Cluster transition sources: {len(self.cluster_transitions)}")

    def _extract_recent_traj_features(self, current_trajectory):
        # Extract all visited POI IDs
        poi_ids = [int(x) for x in re.findall(r'visit POI ID (\d+)', current_trajectory)]
        if not poi_ids:
            return None

        # Extract hours (At yyyy-mm-dd hh:)
        hours = [int(h) for h in re.findall(r'At \d{4}-\d{2}-\d{2} (\d{2}):', current_trajectory)]
        if len(hours) < len(poi_ids):
            hours = hours + [0] * (len(poi_ids) - len(hours))
        else:
            hours = hours[:len(poi_ids)]

        # Query features from database using ID lookup
        lats = []
        lons = []
        cats = []
        for pid in poi_ids:
            if pid in self.poi_id_to_idx:
                idx = self.poi_id_to_idx[pid]
                lats.append(float(self.poi_lats[idx]))
                lons.append(float(self.poi_lons[idx]))
                cats.append(str(self.poi_categories[idx]))
            else:
                lats.append(0.0)
                lons.append(0.0)
                cats.append("Unknown")

        return {
            "pois": poi_ids,
            "lats": np.array(lats),
            "lons": np.array(lons),
            "cats": cats,
            "hours": hours,
        }

    def _calculate_hsid_sim(self, pid1_str, pid2_str):
        if not self.hsid_data:
            return 0.0
        h1 = self.hsid_data.get(pid1_str)
        h2 = self.hsid_data.get(pid2_str)
        if not h1 or not h2:
            return 0.0
        
        sim = 0.0
        if h1.get("coarse_region") == h2.get("coarse_region"):
            sim += 0.1
        if h1.get("fine_region") == h2.get("fine_region"):
            sim += 0.3
        if h1.get("category") == h2.get("category"):
            sim += 0.2
        if h1.get("semantic_cluster") == h2.get("semantic_cluster"):
            sim += 0.4
        return sim

    def get_expert_candidates(self, current_trajectory, query_embedding, k=25):
        """
        Retrieves next POI candidates using a robust Two-Stage Retrieval-Reranking framework.
        
        1. Retrieval Stage:
           Generates a candidate pool using Spatial (top 100 nearest) and Semantic (top 100 cosine sim) routes.
           Filters out any candidates with distance > 50km.
           \[
           \mathcal{P}_{\text{candidate}} = \{ p \in \mathcal{P}_{\text{spatial}} \cup \mathcal{P}_{\text{semantic}} \mid \text{dist}(p) \le 50\text{km} \}
           \]
           
        2. Reranking Stage:
           Aligns all 4 experts (spatial, transition, temporal, semantic) into Reciprocal Rank space:
           \[
           Score_e(p) = \frac{1}{\mu + R_e(p)}
           \]
           Computes weighted fusion based on Heuristic Router weights \(w_e\):
           \[
           Score_{\text{final}}(p) = \sum_{e \in \mathcal{E}} w_e \cdot Score_e(p)
           \]
        """
        traj_feat = self._extract_recent_traj_features(current_trajectory)
        if not traj_feat:
            # Fallback to standard baseline if trajectory parsing fails
            return [int(x["poi_id"]) for x in self.rag_finder.search_similar_pois(f"User trajectory: {current_trajectory}", k=k)]

        last_poi = traj_feat["pois"][-1]
        last_lat = traj_feat["lats"][-1]
        last_lon = traj_feat["lons"][-1]
        last_cat = traj_feat["cats"][-1]
        last_hour = traj_feat["hours"][-1]

        # Calculate distances from last POI to all POIs
        distances = haversine_distance(last_lat, last_lon, self.poi_lats, self.poi_lons)

        # ----------------------------------------------------
        # 1. RETRIEVAL STAGE (Dual-route candidate generation)
        # ----------------------------------------------------
        # Spatial Route: top 100 nearest POIs (unfiltered first to mimic Eng-Opt)
        spatial_indices = np.argsort(distances)[:100]
        spatial_list = [int(idx) for idx in spatial_indices]

        # Semantic Route: top 100 vector-similar POIs
        if hasattr(self.rag_finder, "faiss_index") and self.rag_finder.faiss_index is not None:
            distances_vec, indices_vec = self.rag_finder.faiss_index.search(
                np.expand_dims(query_embedding, axis=0), 100
            )
            semantic_list = [int(idx) for idx in indices_vec[0] if idx >= 0]
            semantic_scores = list(distances_vec[0])
        else:
            semantic_scores_all = self.rag_finder.embeddings @ query_embedding
            semantic_list = list(np.argsort(semantic_scores_all)[::-1][:100])
            semantic_scores = list(semantic_scores_all[semantic_list])

        # Integrate HSID similarity directly into semantic list rank ordering
        combined_semantic_scores = []
        for idx_i, idx in enumerate(semantic_list):
            pid_str = str(self.poi_ids[idx])
            sim = 0.0
            if len(traj_feat["pois"]) >= 1:
                last_pid_str = str(traj_feat["pois"][-1])
                sim += 1.0 * self._calculate_hsid_sim(last_pid_str, pid_str)
                if len(traj_feat["pois"]) >= 2:
                    prev_pid_str = str(traj_feat["pois"][-2])
                    sim += 0.5 * self._calculate_hsid_sim(prev_pid_str, pid_str)
            # Combine cosine similarity and HSID similarity (gamma = 0.5)
            combined_score = float(semantic_scores[idx_i]) + 0.5 * sim
            combined_semantic_scores.append((idx, combined_score))
        
        combined_semantic_scores.sort(key=lambda x: x[1], reverse=True)
        semantic_list = [item[0] for item in combined_semantic_scores]

        # Merge candidate pool and apply hard filter (exclude > 50km)
        candidates_pool = list(set(spatial_list + semantic_list))
        candidates_pool = [idx for idx in candidates_pool if distances[idx] <= 50.0]

        if not candidates_pool:
            candidates_pool = spatial_list  # Fallback

        # ----------------------------------------------------
        # 2. EXPERT SCORING (Feature extraction on pool)
        # ----------------------------------------------------
        # (1) Transition Scores
        transition_scores = np.zeros(len(self.poi_ids), dtype=np.float32)
        if last_poi in self.poi_transitions:
            for next_p, count in self.poi_transitions[last_poi].items():
                if next_p in self.poi_id_to_idx:
                    idx = self.poi_id_to_idx[next_p]
                    transition_scores[idx] += 3.0 * np.log1p(count)
                    
        if last_cat in self.cat_transitions:
            for next_c, count in self.cat_transitions[last_cat].items():
                matching_indices = np.where(self.poi_categories == next_c)[0]
                transition_scores[matching_indices] += 0.2 * np.log1p(count)

        # Region/Cluster transitions on candidates_pool
        last_poi_str = str(last_poi)
        if last_poi_str in self.hsid_data:
            last_h = self.hsid_data[last_poi_str]
            last_coarse = last_h.get("coarse_region")
            last_fine = last_h.get("fine_region")
            last_cluster = last_h.get("semantic_cluster")

            if last_coarse and last_coarse in self.coarse_transitions:
                for next_coarse, count in self.coarse_transitions[last_coarse].items():
                    for idx in candidates_pool:
                        pid_str = str(self.poi_ids[idx])
                        if pid_str in self.hsid_data and self.hsid_data[pid_str].get("coarse_region") == next_coarse:
                            transition_scores[idx] += 0.1 * np.log1p(count)

            if last_fine and last_fine in self.fine_transitions:
                for next_fine, count in self.fine_transitions[last_fine].items():
                    for idx in candidates_pool:
                        pid_str = str(self.poi_ids[idx])
                        if pid_str in self.hsid_data and self.hsid_data[pid_str].get("fine_region") == next_fine:
                            transition_scores[idx] += 0.3 * np.log1p(count)

            if last_cluster and last_cluster in self.cluster_transitions:
                for next_cluster, count in self.cluster_transitions[last_cluster].items():
                    for idx in candidates_pool:
                        pid_str = str(self.poi_ids[idx])
                        if pid_str in self.hsid_data and self.hsid_data[pid_str].get("semantic_cluster") == next_cluster:
                            transition_scores[idx] += 0.4 * np.log1p(count)

        # (2) Temporal Scores
        temporal_scores = np.zeros(len(self.poi_ids), dtype=np.float32)
        hour_probs = self.hourly_cat_probs.get(last_hour, {})
        if hour_probs:
            for cat, prob in hour_probs.items():
                matching_indices = np.where(self.poi_categories == cat)[0]
                temporal_scores[matching_indices] += float(prob)

        # Apply strong geographic constraints on non-core scores (decay for dist > 15km)
        for idx in candidates_pool:
            dist = distances[idx]
            if dist > 15.0:
                decay = np.exp(-(dist - 15.0) / 10.0)
                transition_scores[idx] *= decay
                temporal_scores[idx] *= decay

        # ----------------------------------------------------
        # 3. HEURISTIC ROUTER (Weights generation)
        # ----------------------------------------------------
        if len(traj_feat["lats"]) >= 2:
            step_distances = haversine_distance(
                traj_feat["lats"][:-1], traj_feat["lons"][:-1],
                traj_feat["lats"][1:], traj_feat["lons"][1:]
            )
            avg_step_dist = np.mean(step_distances)
            repeat_rate = 1.0 - (len(set(traj_feat["pois"])) / len(traj_feat["pois"]))
        else:
            avg_step_dist = 5.0
            repeat_rate = 0.0

        # Dynamic weights definition
        w_spatial = 3.5
        w_transition = 0.0
        w_temporal = 0.0
        w_semantic = 3.0

        if avg_step_dist < 1.5:
            w_spatial += 0.5
            if repeat_rate > 0.2:
                w_transition += 0.3
        elif avg_step_dist > 15.0:
            w_semantic += 0.5
            w_spatial -= 0.5
            w_temporal += 0.2

        if repeat_rate > 0.4:
            w_transition += 0.5

        # L1 normalize weights
        weights = np.array([w_spatial, w_transition, w_temporal, w_semantic], dtype=np.float32)
        weights /= np.sum(weights)

        # ----------------------------------------------------
        # 4. RERANKING (Weighted Reciprocal Rank Fusion)
        # ----------------------------------------------------
        rank_spatial = {idx: r for r, idx in enumerate(spatial_list)}
        rank_semantic = {idx: r for r, idx in enumerate(semantic_list)}

        # Rank transition scores inside candidate pool (only for high-confidence transitions >= 0.5)
        pool_transition_scores = [transition_scores[idx] for idx in candidates_pool]
        sorted_trans_pool = [candidates_pool[i] for i in np.argsort(pool_transition_scores)[::-1]]
        rank_transition = {}
        r_count = 0
        for idx in sorted_trans_pool:
            if transition_scores[idx] >= 0.5:
                rank_transition[idx] = r_count
                r_count += 1

        # Rank temporal scores inside candidate pool (only for high-confidence temporal probs >= 0.05)
        pool_temporal_scores = [temporal_scores[idx] for idx in candidates_pool]
        sorted_temp_pool = [candidates_pool[i] for i in np.argsort(pool_temporal_scores)[::-1]]
        rank_temporal = {}
        r_count = 0
        for idx in sorted_temp_pool:
            if temporal_scores[idx] >= 0.05:
                rank_temporal[idx] = r_count
                r_count += 1

        mu = 50.0
        final_scores = []
        for idx in candidates_pool:
            score_sp = 1.0 / (mu + rank_spatial[idx]) if idx in rank_spatial else 0.0
            score_se = 1.0 / (mu + rank_semantic[idx]) if idx in rank_semantic else 0.0
            score_tr = 1.0 / (mu + rank_transition[idx]) if idx in rank_transition else 0.0
            score_te = 1.0 / (mu + rank_temporal[idx]) if idx in rank_temporal else 0.0

            total_score = (
                weights[0] * score_sp +
                weights[3] * score_se +
                weights[1] * score_tr +
                weights[2] * score_te
            )
            final_scores.append((idx, total_score))

        final_scores.sort(key=lambda x: x[1], reverse=True)
        final_poi_ids = [int(self.poi_ids[item[0]]) for item in final_scores]
        return final_poi_ids[:k]
