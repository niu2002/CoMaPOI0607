import json
import re

def format_historical_info(historical_dist):
    if not historical_dist:
        return "None"
    
    # If it is a string already, just return it
    if isinstance(historical_dist, str):
        return historical_dist
        
    # If it is a dictionary containing historical_information
    if isinstance(historical_dist, dict):
        hist_info = historical_dist.get("historical_information", None)
        if hist_info is None:
            try:
                return json.dumps(historical_dist, ensure_ascii=False)
            except Exception:
                return str(historical_dist)
        historical_dist = hist_info
        
    # If historical_dist is a list (e.g. of dictionaries representing trajectories)
    if isinstance(historical_dist, list):
        checkins = []
        pattern = r"At \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}, [A-Za-z]+, user \d+ visit POI ID \d+ \([^)]+\)\.?"
        for item in historical_dist:
            if isinstance(item, dict) and "messages" in item:
                for msg in item["messages"]:
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        content = msg.get("content", "")
                        for m in re.findall(pattern, content):
                            m_clean = m.strip()
                            if m_clean.endswith("."):
                                m_clean = m_clean[:-1].strip()
                            checkins.append(m_clean)
            elif isinstance(item, str):
                for m in re.findall(pattern, item):
                    m_clean = m.strip()
                    if m_clean.endswith("."):
                        m_clean = m_clean[:-1].strip()
                    checkins.append(m_clean)
        
        if checkins:
            # Sort chronologically by the timestamp prefix
            sorted_checkins = sorted(checkins)
            return "\n".join(sorted_checkins)
            
    # Fallback to json.dumps
    try:
        return json.dumps(historical_dist, ensure_ascii=False)
    except Exception:
        return str(historical_dist)

class Forwar_prompter:
    def __init__(self, args, user_id, subtrajectory_id, current_trajectory, next_poi_info):
        self.user_id = user_id
        self.subtrajectory_id = subtrajectory_id
        self.current_trajectory = current_trajectory
        self.args = args
        self.next_poi_info = next_poi_info

    def get_a1p1_prompt(self, historical_distribution):
        formatted_history = format_historical_info(historical_distribution)
        prompt_data = f"""
        ###IDENTITY and PURPOSE:
        You are an expert User Profiler specialized in constructing long-term user profiles based on the user's next POI visit information.
        ###TASK:
        For User ID: {self.user_id}, Subtrajectory ID: {self.subtrajectory_id}, use the provided next POI information and historical trajectory distribution to generate a long-term profile that is highly related to this POI.
        ###True Next POI Information:
        - POI ID: {self.next_poi_info[0]}
        - Category: {self.next_poi_info[1]}
        - Locations: Latitude: {self.next_poi_info[2]}, Longitude: {self.next_poi_info[3]}

        ###historical trajectory data: {formatted_history}"""

        return json.dumps(prompt_data, indent=2)

    def get_a1p2_prompt(self, long_term_profile):
        prompt_data = f"""
    ###IDENTITY and PURPOSE:
    You are an expert User Profiler, specialized in generating candidate POI IDs based on the long-term profile, ensuring the provided next POI ID is ranked first.

    ###TASK:
    For User ID: {self.user_id}, Subtrajectory ID: {self.subtrajectory_id}, use the provided long-term profile and current trajectory to generate 25 best candidate POI IDs, with the next POI ID ranked first.


    ###CURRENT TRAJECTORY:
    {self.current_trajectory}

    ###Long-Term Profile:
    {long_term_profile}

    ###True Next POI Information:
    - POI ID: {self.next_poi_info[0]}
    - Category: {self.next_poi_info[1]}
    - Locations:
        - Latitude: {self.next_poi_info[2]}
        - Longitude: {self.next_poi_info[3]}

    ###STEPS:
    1. Candidate Generation: Generate an initial list of POI IDs that align with the long-term profile.
    2. Include the provided next POI ID and ensure it is ranked first.
    3. Candidate Filtering: Remove POIs that are unlikely to be relevant based on the long-term profile and current trajectory.
    4. Candidate Sorting: Place the next POI ID first, followed by other relevant POIs.
    5. Final List: Provide a sorted list of exactly 25 unique candidate POI IDs, ensuring no duplicates.

    ###Description:
    Ensure the response adheres to the following criteria:
    - The next POI ID is the first item in the list.
    - Values must fall within the range [1, {self.args.max_item}].
    - The list must contain exactly 25 unique POI IDs with no duplicates.
    """
        return prompt_data

    def get_a2p1_prompt(self):
        prompt_data = f"""
    ###IDENTITY and PURPOSE:
    You are an expert User Profiler specialized in constructing recent mobility patterns based on the user's next POI visit information.

    ###TASK:
    For User ID: {self.user_id}, Subtrajectory ID: {self.subtrajectory_id}, use the provided next POI information, current trajectory, and its distribution to generate a recent mobility pattern description that is highly related to this POI.

    ###True Next POI Information:
    - POI ID: {self.next_poi_info[0]}
    - Category: {self.next_poi_info[1]}
    - Locations:
        - Latitude: {self.next_poi_info[2]}
        - Longitude: {self.next_poi_info[3]}

    ###Current Trajectory:
    {self.current_trajectory}
    ###STEPS:
    1. Temporal Analysis: Align the user's activity times with the next POI's active periods.
    2. Categorical Analysis: Emphasize categories matching the next POI.
    3. Geographical Analysis: Focus on areas near the next POI's location.
    4. Periodic Analysis: Highlight patterns that logically lead to visiting the next POI.
    5. Summary: Combine the analyses to describe the user's recent preferences and mobility trends related to the next POI.
    """
        return prompt_data

    def get_a2p2_prompt(self, mobility_description, similar_poi_list):
        prompt_data = f"""
    IDENTITY and PURPOSE:
    You are an expert POI Forecaster, specialized in generating candidate POI IDs based on recent mobility patterns, ensuring the provided next POI ID is ranked first.

    TASK:
    For user_{self.user_id}, use the provided inputs to generate 25 best candidate POI IDs, with the next POI ID ranked first.

    User:
    User ID: {self.user_id}, Subtrajectory ID: {self.subtrajectory_id}, indicating that this current trajectory is the {self.subtrajectory_id}th subtrajectory of user {self.user_id}.

    INPUTS:
    - Current Trajectory: {self.current_trajectory}
    - Mobility Pattern Description: {mobility_description}
    - Similar POI List: {similar_poi_list}

    True Next POI Information:
    - POI ID: {self.next_poi_info[0]}
    - Category: {self.next_poi_info[1]}
    - Locations:
        - Latitude: {self.next_poi_info[2]}
        - Longitude: {self.next_poi_info[3]}

    STEPS:
    1. Data Integration: Combine the provided inputs to understand the user's recent behavior.
    2. Candidate Generation: Generate a list of POI IDs that align with the mobility pattern and similar POIs.
    3. Include the provided next POI ID and ensure it is ranked first.
    4. Candidate Filtering: Exclude POIs unlikely to be relevant.
    5. Candidate Sorting: Place the next POI ID first, followed by other relevant POIs.
    6. Ensure the predicted IDs are unique and within the range [1, {self.args.max_item}].
    7. Final List: Provide a sorted list of exactly 25 unique candidate POI IDs, ensuring no duplicates.

    """
        return prompt_data

    def get_a3p1_prompt(self, long_term_profile, short_term_profile, candidate_poi_list_agent1,
                        candidate_poi_list_agent2):
        prompt_data = f"""
    IDENTITY and PURPOSE:
    You are an advanced POI ID Predictor tasked with accurately predicting the next POI ID for user_{self.user_id} based on their trajectory and profiles.

    TASK:
    Predict the most likely next POI ID for the user (User ID: {self.user_id}, Subtrajectory ID: {self.subtrajectory_id}), considering their current trajectory, long-term profile, and short-term profile.

    INPUTS:
    - Current Trajectory: {self.current_trajectory}
    - Long-Term Profile: {long_term_profile}
    - Short-Term Profile: {short_term_profile}
    - Candidates from Long-Term Profile: {candidate_poi_list_agent1}
    - Candidates from Short-Term Profile: {candidate_poi_list_agent2}

    STEPS:
    1. Candidate Analysis: Analyze the user's trajectory and profiles to identify relevant POIs.
    2. Prediction: Output the predicted next POI ID. 
    3. Validation: Ensure the selected POI ID is consistent with the user's context and trajectory.
    """
        return prompt_data




class Inverse_prompter:
    def __init__(self, args, user_id, subtrajectory_id, current_trajectory, next_poi_info):
        self.user_id = user_id
        self.subtrajectory_id = subtrajectory_id
        self.current_trajectory = current_trajectory
        self.args = args
        self.next_poi_info = next_poi_info

    def get_a1p1_prompt(self, historical_distribution):
        formatted_history = format_historical_info(historical_distribution)
        system_prompt_format = """system: Respond with a JSON dictionary in a markdown's fenced code block as follows:\n```json\n{\n  "historical_distribution": ["A detailed textual analysis of the user's recent mobility patterns"]\n}\n```"""

        prompt_data = {
            "IDENTITY and PURPOSE": "You are an expert User Profiler specialized in constructing long-term user profiles based on the user's next POI visit information.",
            "TASK": f"For user_{self.user_id}, use the provided next POI information and historical trajectory distribution to generate a long-term profile that is highly related to this POI.",
            "User": f"User ID:{self.user_id}, Subtrajectory ID:{self.subtrajectory_id}, indicating that this current trajectory is the {self.subtrajectory_id}th subtrajectory of user {self.user_id}.",
            "True Next POI Information": {
                "poi_id": self.next_poi_info[0],
                "category": self.next_poi_info[1],
                "locations": {
                    "latitude": self.next_poi_info[2],
                    "longitude": self.next_poi_info[3]}},
            "INPUT": f"User's historical trajectory data: {formatted_history}",
            "STEPS": [
                "Analyze the characteristics of the next POI (e.g., category, location).",
                "Construct a long-term profile that aligns with these characteristics and logically leads to the user visiting this next POI.",
                "Ensure the profile reflects consistent patterns and preferences related to the next POI."
            ],
            "OUTPUT FORMAT": system_prompt_format
        }

        return json.dumps(prompt_data, indent=2)

    def get_a1p2_prompt(self, long_term_profile):
        system_prompt_format = """system: Respond a JSON dictionary in a markdown's fenced code block as follows:\njson\n{"next_poi_id": ["value1", "value2", ..., "value25"]}\n'}]"""
        prompt_data = {
        "IDENTITY and PURPOSE": "You are an expert User Profiler, specialized in generating candidate POI IDs based on the long-term profile, ensuring the provided next POI ID is ranked first.",
        "TASK": f"For user_{self.user_id}, use the provided long-term profile and current trajectory to generate 25 best candidate POI IDs, with the next POI ID ranked first.",
        "User": f"User ID:{self.user_id}, Subtrajectory ID:{self.subtrajectory_id}, indicating that this current trajectory is the {self.subtrajectory_id}th subtrajectory of user {self.user_id}.",
        "CURRENT TRAJECTORY": self.current_trajectory,
        "Long-Term Profile": long_term_profile,
        "True Next POI Information": {
            "poi_id": self.next_poi_info[0],
            "category": self.next_poi_info[1],
            "locations": {
                "latitude": self.next_poi_info[2],
                "longitude": self.next_poi_info[3]}},
        "STEPS": [
            "Candidate Generation: Generate an initial list of POI IDs that align with the long-term profile.",
            "Include the provided next POI ID and ensure it is ranked first.",
            "Candidate Filtering: Remove POIs that are unlikely to be relevant based on the long-term profile and current trajectory.",
            "Candidate Sorting: Place the next POI ID first, followed by other relevant POIs.",
            "Final List: Provide a sorted list of exactly 25 unique candidate POI IDs, ensuring no duplicates."
        ],
        "OUTPUT FORMAT": {
            "Description": "Ensure the response adheres to the following criteria:",
            "Criteria": [
                "The response is a valid JSON dictionary that can be parsed with `json.loads`.",
                "The next POI ID is the first item in the list.",
                "Each POI ID is a unique integer or string within the specified range.",
                f"Values must fall within the range [1, {self.args.max_item}].",
                "The list must contain exactly 25 unique POI IDs with no duplicates.",
                "No nested structures or invalid characters are allowed in the response."
            ],
            "Example Format": system_prompt_format
        }}
        return json.dumps(prompt_data, indent=2)

    def get_a2p1_prompt(self):
        system_prompt_format = """system: Respond with a JSON dictionary in a markdown's fenced code block as follows:\n```json\n{\n  "recent_mobility_analysis": ["A detailed textual analysis of the user's recent mobility patterns"]\n}\n```"""
        prompt_data = {
        "IDENTITY and PURPOSE": "You are an expert User Profiler specialized in constructing recent mobility patterns based on the user's next POI visit information.",
        "TASK": f"For user_{self.user_id}, use the provided next POI information, current trajectory, and its distribution to generate a recent mobility pattern description that is highly related to this POI.",
        "User": f"User ID:{self.user_id}, Subtrajectory ID:{self.subtrajectory_id}, indicating that this current trajectory is the {self.subtrajectory_id}th subtrajectory of user {self.user_id}.",
        "True Next POI Information": {
            "poi_id": self.next_poi_info[0],
            "category": self.next_poi_info[1],
            "locations": {
                "latitude": self.next_poi_info[2],
                "longitude": self.next_poi_info[3]}},
        "INPUT": {
            "Current Trajectory": self.current_trajectory,
        },
        "STEPS": [
            "Temporal Analysis: Align the user's activity times with the next POI's active periods.",
            "Categorical Analysis: Emphasize categories matching the next POI.",
            "Geographical Analysis: Focus on areas near the next POI's location.",
            "Periodic Analysis: Highlight patterns that logically lead to visiting the next POI.",
            "Summary: Combine the analyses to describe the user's recent preferences and mobility trends related to the next POI."
        ],
        "OUTPUT FORMAT": system_prompt_format}
        return json.dumps(prompt_data, indent=2)

    def get_a2p2_prompt(self, mobility_description, similar_poi_list):
        system_prompt_format = """system: Respond with a JSON dictionary in a markdown's fenced code block as follows:\n```json\n{\n  "next_poi_id": ["value1", "value2", ..., "value25"]\n}\n```"""
        prompt_data = {
        "IDENTITY and PURPOSE": "You are an expert POI Forecaster, specialized in generating candidate POI IDs based on recent mobility patterns, ensuring the provided next POI ID is ranked first.",
        "TASK": f"For user_{self.user_id}, use the provided inputs to generate 25 best candidate POI IDs, with the next POI ID ranked first.",
        "User": f"User ID:{self.user_id}, Subtrajectory ID:{self.subtrajectory_id}, indicating that this current trajectory is the {self.subtrajectory_id}th subtrajectory of user {self.user_id}.",
        "INPUTS": {
            "Current Trajectory": self.current_trajectory,
            "Mobility Pattern Description": mobility_description,
            "Similar POI List": similar_poi_list
        },
        "True Next POI Information": {
            "poi_id": self.next_poi_info[0],
            "category": self.next_poi_info[1],
            "locations": {
                "latitude": self.next_poi_info[2],
                "longitude": self.next_poi_info[3]}},
        "STEPS": [
            "Data Integration: Combine the provided inputs to understand the user's recent behavior.",
            "Candidate Generation: Generate a list of POI IDs that align with the mobility pattern and similar POIs.",
            "Include the provided next POI ID and ensure it is ranked first.",
            "Candidate Filtering: Exclude POIs unlikely to be relevant.",
            "Candidate Sorting: Place the next POI ID first, followed by other relevant POIs.",
            f"Ensure the predicted IDs are unique and within the range [1, {self.args.max_item}].",
            "Final List: Provide a sorted list of exactly 25 unique candidate POI IDs, ensuring no duplicates."
        ],
        "OUTPUT FORMAT": system_prompt_format}
        return json.dumps(prompt_data, indent=2)

    def get_a3p1_prompt(self, long_term_profile, short_term_profile, candidate_poi_list_agent1,
                        candidate_poi_list_agent2):
        system_prompt_format = """system: Respond a JSON dictionary in a markdown's fenced code block as follows:\n```json\n{\n  "next_poi_id": "value"\n}\n```"""
        prompt_data = {
            "IDENTITY and PURPOSE": f"You are an advanced POI ID Predictor tasked with accurately predicting the next POI ID for user_{self.user_id} based on their trajectory and profiles.",
            "TASK": f"Predict the most likely next POI ID for the user, considering their current trajectory, long-term profile, and short-term profile.",
            "User": f"User ID:{self.user_id}, Subtrajectory ID:{self.subtrajectory_id}, indicating that this current trajectory is the {self.subtrajectory_id}th subtrajectory of user {self.user_id}.",
            "INPUTS": {
                "Current Trajectory": self.current_trajectory,
                "Long-Term Profile": long_term_profile,
                "Short-Term Profile": short_term_profile,
                "Candidates from Long-Term Profile": candidate_poi_list_agent1,
                "Candidates from Short-Term Profile": candidate_poi_list_agent2
            },
            "STEPS": [
                "Candidate Analysis: Analyze the user's trajectory and profiles to identify relevant POIs.",
                "Prediction: Select the most likely next POI ID from the candidates.",
                "Validation: Ensure the selected POI ID is consistent with the user's context and trajectory."
            ],
            "OUTPUT FORMAT": {
                "Description": "Provide the predicted next POI ID.",
                "Criteria": [
                    "The response is a valid JSON dictionary parsable by `json.loads`.",
                    "Contains exactly one key-value pair: `next_poi_id`.",
                    "The `next_poi_id` must be a unique integer or string within the specified range.",
                    f"The `next_poi_id` must fall within the range [1, {self.args.max_item}].",
                    "No nested structures or invalid characters are allowed."
                ],
                "Example Format": system_prompt_format
            }
        }
        return json.dumps(prompt_data, indent=2)
