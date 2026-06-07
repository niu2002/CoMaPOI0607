import json
import pandas as pd


class PromptProvider:
    def __init__(self, args, user_id, current_trajectory):
        self.user_id = user_id
        self.current_trajectory = current_trajectory
        self.args = args

    @staticmethod
    def _json_block(schema: str) -> str:
        return f"```json\n{schema}\n```"

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
                "Time Distribution Analysis: Identify active time periods (e.g., morning, afternoon, evening).",
                "Spatial Distribution Analysis: Determine frequently visited geographic areas (e.g., city center, suburbs).",
                "Category Distribution Analysis: Analyze the distribution of POI categories visited (e.g., restaurants, supermarkets, entertainment venues).",
                "Summary: Summarize the user's long-term profile based on the analyses above."
            ],
            "IMPORTANT": self._json_only_rules("historical_profile") + [
                "The profile text must stay within 500 tokens.",
                "Put the full profile summary in a single string value, not a list.",
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
                "1. Analyze the time pattern in the current trajectory (e.g., time of day, day of week).",
                "2. Analyze the spatial pattern (e.g., geographic area, distance between POIs).",
                "3. Analyze the category pattern (e.g., types of POIs visited).",
                "4. Identify any specific needs or intentions based on the trajectory (e.g., shopping, dining, entertainment).",
                "5. Summarize the user's short-term mobility profile based on the analyses above."
            ],
            "IMPORTANT": self._json_only_rules("current_profile") + [
                "The profile text must stay within 500 tokens.",
                "Put the full profile summary in a single string value, not a list.",
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

    def get_a3p1_prompt(self, long_term_profile, short_term_profile, candidate_poi_list_agent1, candidate_poi_list_agent2):
        id_list = [f"\"{i + 1}th unique ID\"" for i in range(self.args.top_k)]
        id_list[0] = "\"best unique ID\""
        id_list_str = ", ".join(id_list)
        system_prompt_format = self._json_block(
            f'{{"next_poi_id": [{id_list_str}]}}'
        )

        prompt_data = {
            "IDENTITY and PURPOSE": "You are an expert POI Predictor specialized in combining insights from long-term user profiles, short-term mobility patterns, and candidate POIs to predict the next POI a user will visit.",
            "TASK": f"For user_{self.user_id}, use the provided long-term profile, short-term mobility profile, and candidate POI lists to predict the top {self.args.top_k} POIs the user is most likely to visit next.",
            "User": f"User ID:{self.user_id}",
            "CURRENT TRAJECTORY": self.current_trajectory,
            "Long-Term Profile": long_term_profile,
            "Short-Term Mobility Profile": short_term_profile,
            "Candidate POIs from Profile Analysis": candidate_poi_list_agent1,
            "Candidate POIs from Mobility Analysis": candidate_poi_list_agent2,
            "STEPS": [
                "1. Analyze the long-term profile to understand the user's general preferences and patterns.",
                "2. Analyze the short-term mobility profile to understand the user's current context and needs.",
                "3. Review both candidate POI lists.",
                "4. Combine insights from all sources to identify the most likely POIs.",
                f"5. Rank and select the top {self.args.top_k} POIs the user is most likely to visit next.",
                "6. Ensure the first POI ID in your list is the most likely one."
            ],
            "IMPORTANT": self._json_only_rules("next_poi_id", exact_count=self.args.top_k) + [
                "Balance long-term preferences with short-term context.",
                "Strongly prioritize POI IDs from the provided candidate lists.",
                "Provide only numeric POI IDs, not years, dates, times, coordinates, ranks, or explanations.",
                "Ensure all IDs are unique positive integers.",
            ],
            "OUTPUT_FORMAT": system_prompt_format,
        }
        return json.dumps(prompt_data, indent=2)

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
