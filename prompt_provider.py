import json
import pandas as pd


class PromptProvider:
    def __init__(self, args, user_id, current_trajectory):
        self.user_id = user_id
        self.args = args
        self.current_trajectory = self._get_compact_trajectory(current_trajectory)

    @staticmethod
    def _json_block(schema: str) -> str:
        return f"```json\n{schema}\n```"

    def _profile_token_limit(self) -> int:
        return int(getattr(self.args, "profile_max_tokens", 220))

    def _json_only_rules(self, key_name: str, exact_count: int | None = None, restrict_to_candidates: bool = False):
        rules = [
            "Return exactly one JSON object in exactly one fenced ```json code block.",
            "Return no prose, no analysis, no bullet points, and no explanation outside the JSON block.",
            f'The JSON object must contain only the key "{key_name}".',
        ]
        if exact_count is not None:
            rules.append(f'The value of "{key_name}" must contain exactly {exact_count} items.')
        if restrict_to_candidates:
            rules.append("Every returned POI ID must be selected from the provided candidate list.")
        return rules

    def get_a1p1_prompt(self, historical_distribution):
        system_prompt_format = self._json_block(
            '{"historical_profile": "A concise long-term profile summary"}'
        )
        prompt_data = {
            "IDENTITY and PURPOSE": "You are an expert User Profiler specialized in constructing long-term user profiles based on the user's historical trajectory data.",
            "TASK": f"For user_{self.user_id}, use the provided historical trajectory distribution to generate a long-term profile that reflects the user's preferences, behavioral patterns, and likely characteristics.",
            "User": f"User ID:{self.user_id}",
            "INPUT": f"User's historical trajectory data: {historical_distribution}",
            "STEPS": [
                "Identify only predictive long-term signals: recurring time windows, stable areas, favorite categories, and repeated high-signal POIs.",
                "Compress the result into a concise profile that helps rank the next POI.",
                "Do not copy raw trajectory dates, coordinates, or step-by-step reasoning."
            ],
            "IMPORTANT": self._json_only_rules("historical_profile") + [
                f"The profile text must stay within {self._profile_token_limit()} tokens.",
                "Put the full profile summary in a single string value, not a list.",
                "Prefer compact clauses separated by semicolons.",
            ],
            "OUTPUT_FORMAT": system_prompt_format,
        }
        return json.dumps(prompt_data, indent=2)

    def get_a1p2_prompt(self, long_term_profile):
        id_list = [f"\"{i + 1}th unique ID\"" for i in range(self.args.num_candidate)]
        id_list[0] = "\"best unique ID\""
        id_list_str = ", ".join(id_list)
        system_prompt_format = self._json_block(
            f'{{"candidate_poi_list_from_profile": [{id_list_str}]}}'
        )
        prompt_data = {
            "IDENTITY and PURPOSE": "You are an expert User Profiler, specialized in analyzing long-term user profiles and current trajectories to predict the next POI ID for a user.",
            "TASK": f"For user_{self.user_id}, use the provided long-term profile and current trajectory to generate {self.args.num_candidate} best candidate POI IDs. Prioritize insights from the user's historical behavior and adapt them to the current trajectory context.",
            "User": f"User ID:{self.user_id}",
            "CURRENT TRAJECTORY": self.current_trajectory,
            "Long-Term Profile": long_term_profile,
            "STEPS": [
                "1. Analyze the long-term profile to identify the user's preferences and behavioral patterns.",
                "2. Analyze the current trajectory to understand the user's current context and needs.",
                "3. Generate candidate POI IDs that align with both the user's long-term preferences and current context.",
                f"4. Rank the candidates and select the top {self.args.num_candidate} most likely POIs the user will visit next.",
                "5. Ensure the first POI ID in your list is the most likely one."
            ],
            "IMPORTANT": self._json_only_rules("candidate_poi_list_from_profile", exact_count=self.args.num_candidate) + [
                "Provide only numeric POI IDs, not years, dates, times, coordinates, ranks, or explanations.",
                "Ensure all IDs are unique positive integers.",
            ],
            "OUTPUT_FORMAT": system_prompt_format,
        }
        return json.dumps(prompt_data, indent=2)

    def get_a2p1_prompt(self):
        system_prompt_format = self._json_block(
            '{"current_profile": "A concise short-term mobility profile summary"}'
        )
        prompt_data = {
            "IDENTITY and PURPOSE": "You are an expert Mobility Pattern Analyzer specialized in analyzing users' recent mobility patterns to understand their current context and needs.",
            "TASK": f"For user_{self.user_id}, analyze the provided current trajectory to generate a short-term mobility profile that captures the user's recent behavior and current context.",
            "User": f"User ID:{self.user_id}",
            "INPUT": f"User's current trajectory: {self.current_trajectory}",
            "STEPS": [
                "1. Identify only predictive short-term signals: latest location trend, time context, category intent, and repeated recent POIs.",
                "2. Compress the result into a concise profile that helps rank the next POI.",
                "3. Do not copy raw trajectory dates, coordinates, or step-by-step reasoning."
            ],
            "IMPORTANT": self._json_only_rules("current_profile") + [
                f"The profile text must stay within {self._profile_token_limit()} tokens.",
                "Put the full profile summary in a single string value, not a list.",
                "Prefer compact clauses separated by semicolons.",
            ],
            "OUTPUT_FORMAT": system_prompt_format,
        }
        return json.dumps(prompt_data, indent=2)

    def get_a2p2_prompt(self, short_pattern_response, rag_candidates):
        id_list = [f"\"{i + 1}th unique ID\"" for i in range(self.args.num_candidate)]
        id_list[0] = "\"best unique ID\""
        id_list_str = ", ".join(id_list)
        system_prompt_format = self._json_block(
            f'{{"refined_candidate_from_rag": [{id_list_str}]}}'
        )

        prompt_data = {
            "IDENTITY and PURPOSE": "You are an expert Mobility Pattern Analyzer specialized in refining candidate POIs based on a user's short-term mobility profile and RAG-retrieved candidates.",
            "TASK": f"For user_{self.user_id}, use the provided short-term mobility profile and RAG-retrieved candidate POIs to generate a refined list of {self.args.num_candidate} candidate POI IDs that the user is most likely to visit next.",
            "User": f"User ID:{self.user_id}",
            "Short-Term Mobility Profile": short_pattern_response,
            "RAG-Retrieved Candidate POIs": rag_candidates,
            "STEPS": [
                "1. Analyze the short-term mobility profile to understand the user's current context and needs.",
                "2. Review the RAG-retrieved candidate POIs.",
                "3. Select and rank POIs that best match the user's current context and needs.",
                f"4. Generate a refined list of {self.args.num_candidate} candidate POI IDs.",
                "5. Ensure the first POI ID in your list is the most likely one."
            ],
            "IMPORTANT": self._json_only_rules(
                "refined_candidate_from_rag",
                exact_count=self.args.num_candidate,
                restrict_to_candidates=True,
            ) + [
                "Use only POI IDs from the provided RAG candidate list.",
                "Provide only numeric POI IDs, not years, dates, times, coordinates, ranks, or explanations.",
                "Ensure all IDs are unique positive integers.",
            ],
            "OUTPUT_FORMAT": system_prompt_format,
        }
        return json.dumps(prompt_data, indent=2)

    def _get_compact_trajectory(self, traj_str, max_len=10):
        if not traj_str:
            return ""
        
        import re
        raw_lines = traj_str.strip().split("\n")
        header = ""
        trajectory_units = []
        
        for line in raw_lines:
            line_str = line.strip()
            if not line_str:
                continue
            if line_str.startswith("<") and line_str.endswith(">"):
                header = line_str
                continue
            
            # Split robustly by '. At 20xx' or similar pattern
            parts = re.split(r"\.\s*(?=At\s+\d{4})", line_str)
            for part in parts:
                p_clean = part.strip()
                if p_clean:
                    if not p_clean.endswith("."):
                        p_clean += "."
                    trajectory_units.append(p_clean)
        
        if len(trajectory_units) <= max_len:
            compact_units = trajectory_units
        else:
            compact_units = trajectory_units[-max_len:]
            
        if header:
            return f"{header}\n" + "\n".join(compact_units)
        return "\n".join(compact_units)

    def get_a3p1_prompt(self, long_term_profile, short_term_profile, candidate_poi_list_agent1, candidate_poi_list_agent2,
                        fused_candidate_poi_list=None):
        poi_metadata_pool = {}
        
        def process_candidate_list(cand_list):
            if not cand_list:
                return []
            cleaned_ids = []
            for item in cand_list:
                if isinstance(item, str) and item.startswith("poi_id:"):
                    match_id = re.match(r"^poi_id:\s*(\w+)", item)
                    if match_id:
                        p_id = match_id.group(1)
                        cleaned_ids.append(int(p_id) if p_id.isdigit() else p_id)
                        
                        match_details = re.search(r"\(([^)]+)\)", item)
                        if match_details:
                            poi_metadata_pool[p_id] = match_details.group(1)
                else:
                    cleaned_ids.append(item)
            return cleaned_ids

        clean_a1 = process_candidate_list(candidate_poi_list_agent1)
        clean_a2 = process_candidate_list(candidate_poi_list_agent2)
        clean_fused = process_candidate_list(fused_candidate_poi_list)

        prompt_data = {
            "IDENTITY and PURPOSE": "You are an expert POI Predictor specialized in combining insights from long-term user profiles, short-term mobility patterns, and candidate POIs to predict the next POI a user will visit.",
            "TASK": f"For user_{self.user_id}, use the provided long-term profile, short-term mobility profile, and candidate POI lists to predict the top {self.args.top_k} POIs the user is most likely to visit next.",
            "User": f"User ID:{self.user_id}",
            "CURRENT TRAJECTORY": self.current_trajectory,
            "Long-Term Profile": long_term_profile,
            "Short-Term Mobility Profile": short_term_profile,
            "Candidate POIs from Profile Analysis": clean_a1,
            "Candidate POIs from Mobility Analysis": clean_a2,
            "Fused Candidate POIs": clean_fused,
            "STEPS": [
                "1. Analyze the long-term profile to understand the user's general preferences and patterns.",
                "2. Analyze the short-term mobility profile to understand the user's current context and needs.",
                "3. Review the profile, mobility, and fused candidate POI lists.",
                "4. Combine insights from all sources to identify the most likely POIs. If fused candidates are provided, use them as the primary candidate pool.",
                f"5. Rank and select the top {self.args.top_k} POIs the user is most likely to visit next.",
                "6. Ensure the first POI ID in your list is the most likely one."
            ],
            "IMPORTANT": self._json_only_rules("next_poi_id", exact_count=self.args.top_k) + [
                "Balance long-term preferences with short-term context.",
                "Strongly prioritize POI IDs from the provided candidate lists, especially the fused candidate list when it is non-empty.",
                "Provide only numeric POI IDs, not years, dates, times, coordinates, ranks, or explanations.",
                "Ensure all IDs are unique positive integers.",
            ],
            "OUTPUT_FORMAT": f"Return exactly one JSON block in format: {{\"next_poi_id\": [best_unique_id, 2nd_unique_id, ...]}} containing exactly {self.args.top_k} unique POI IDs.",
        }

        if poi_metadata_pool:
            prompt_data["POI Metadata Pool"] = poi_metadata_pool

        return json.dumps(prompt_data, ensure_ascii=False)

    def agent_retry_prompt(self, invalid_poi_ids):
        id_list = [f"\"{i + 1}th unique ID\"" for i in range(self.args.top_k)]
        id_list[0] = "\"best unique ID\""
        id_list_str = ", ".join(id_list)
        system_prompt_format = self._json_block(
            f'{{"next_poi_id": [{id_list_str}]}}'
        )

        prompt_data = {
            "IDENTITY and PURPOSE": "You are an expert POI Predictor specialized in generating valid POI predictions.",
            "TASK": f"The previous prediction contained invalid POI IDs. Please generate a new list of {self.args.top_k} valid POI IDs.",
            "Previous Invalid Prediction": invalid_poi_ids,
            "REQUIREMENTS": self._json_only_rules("next_poi_id", exact_count=self.args.top_k) + [
                f"All IDs must be positive integers between 1 and {self.args.max_item}.",
                "All IDs must be unique.",
                "The first ID should be the most likely POI the user will visit next.",
            ],
            "OUTPUT_FORMAT": system_prompt_format,
        }
        return json.dumps(prompt_data, indent=2)
