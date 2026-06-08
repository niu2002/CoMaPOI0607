import re
import json
import logging

def _iter_json_candidates(content):
    if isinstance(content, dict):
        yield content
        return

    if not isinstance(content, str):
        return

    text = content.strip()
    if text:
        yield text

    for block in re.findall(r'```(?:json)?\s*([\s\S]*?)\s*```', text, flags=re.IGNORECASE):
        block = block.strip()
        if block:
            yield block


def extract_json_field(content, key_name):
    for candidate in _iter_json_candidates(content):
        if isinstance(candidate, dict):
            data = candidate
        else:
            try:
                data = json.loads(candidate)
            except (json.JSONDecodeError, TypeError):
                continue

        if isinstance(data, dict) and key_name in data:
            return data[key_name]

    if isinstance(content, str):
        list_pattern = rf'"{re.escape(key_name)}"\s*:\s*\[([^\]]*)\]'
        list_match = re.search(list_pattern, content, flags=re.DOTALL)
        if list_match:
            raw_items = list_match.group(1).strip()
            if not raw_items:
                return []
            return [item.strip() for item in re.split(r',\s*', raw_items) if item.strip()]

        str_pattern = rf'"{re.escape(key_name)}"\s*:\s*"([\s\S]*?)"'
        str_match = re.search(str_pattern, content, flags=re.DOTALL)
        if str_match:
            return str_match.group(1).strip()

    return None


def extract_predicted_pois(content, top_k, key_name='next_poi_id', strict=False):
    """
    Extract 'next_poi_id' list from response content.
    Returns at most top_k POI IDs, filtering out non-numeric data.

    Args:
        content: Response content to parse
        top_k: Maximum number of POIs to return
        key_name: Key name to extract (default: 'next_poi_id')

    Returns:
        list: List of POI IDs
    """
    poi_ids = []
    if isinstance(content, list):
        for poi_id in content:
            normalized = _normalize_poi_value(poi_id)
            if normalized is not None:
                poi_ids.append(normalized)
        return poi_ids[:top_k]

    extracted_value = extract_json_field(content, key_name)
    if extracted_value is not None:
        if not isinstance(extracted_value, list):
            extracted_value = [extracted_value]

        for poi_id in extracted_value:
            normalized = _normalize_poi_value(poi_id)
            if normalized is not None:
                poi_ids.append(normalized)

        return poi_ids[:top_k]

    if strict:
        return []

    if isinstance(content, str):
        numbers = re.findall(r'\b\d+\b', content)
        return numbers[:top_k]

    # If all methods fail, return empty list
    return poi_ids[:top_k]


def _normalize_poi_value(value):
    """Normalize one JSON value into a POI ID string without accepting placeholders."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return str(int(value))
    if not isinstance(value, str):
        return None

    text = value.strip()
    if text.isdigit():
        return text

    if re.match(r'^\d{4}-\d{1,2}-\d{1,2}\b', text) or re.match(r'^\d{1,2}:\d{2}(?::\d{2})?\b', text):
        return None

    if re.search(r'\b\d+(?:st|nd|rd|th)\s+unique\s+ID\b', text, flags=re.IGNORECASE):
        return None

    leading_id = re.match(r'^\s*(\d+)\s*(?:$|\(|,|:)', text)
    if leading_id:
        return leading_id.group(1)

    poi_match = re.search(r'\b(?:POI\s*)?ID\s*[:#-]?\s*(\d+)\b', text, flags=re.IGNORECASE)
    if poi_match:
        return poi_match.group(1)

    # Accept strings like "POI 123" or "#123" only when they contain a single number.
    numbers = re.findall(r'\b\d+\b', text)
    if len(numbers) == 1 and re.search(r'\bPOI\b|#', text, flags=re.IGNORECASE):
        return numbers[0]

    return None


def extract_json_from_markdown(text):
    """
    Extract JSON content from markdown code blocks.

    Args:
        text: Markdown text containing code blocks

    Returns:
        str: Extracted JSON content or original text if no code blocks found
    """
    # Look for markdown code blocks (```json ... ```)
    pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    matches = re.findall(pattern, text)

    if matches:
        # Return the content of the first code block
        return matches[0].strip()

    # If no code blocks found, return the original text
    return text


def extract_poi_ids_from_text(text, top_k=10):
    """
    Extract POI IDs from text using various methods.

    Args:
        text: Text to extract POI IDs from
        top_k: Maximum number of POIs to return

    Returns:
        list: List of POI IDs
    """
    poi_ids = []

    # Try to extract JSON from markdown
    json_text = extract_json_from_markdown(text)

    # Try to parse as JSON
    try:
        data = json.loads(json_text)
        if isinstance(data, dict) and 'next_poi_id' in data:
            # Extract POI IDs from next_poi_id field
            for poi in data['next_poi_id']:
                if isinstance(poi, (int, float)):
                    poi_ids.append(str(int(poi)))
                elif isinstance(poi, str):
                    # Extract numeric part
                    match = re.search(r'\b(\d+)\b', poi)
                    if match:
                        poi_ids.append(match.group(1))
    except json.JSONDecodeError:
        # If JSON parsing fails, try to extract numbers directly
        numbers = re.findall(r'\b\d+\b', text)
        poi_ids = numbers

    return poi_ids[:top_k]
