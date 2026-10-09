from typing import Optional, Union, Sequence, Any
from agentscope.agents import DictDialogAgent, ReActAgent, DialogAgent
from agentscope.exception import ResponseParsingError
from agentscope.agents import AgentBase
from agentscope.message import Msg
try:
    from agentscope.parsers import RegexTaggedContentParser, ParserBase
except (ImportError, ModuleNotFoundError):
    class ParserBase:
        pass
    class RegexTaggedContentParser(ParserBase):
        def __init__(self, *args, **kwargs):
            pass
from agentscope.service import (
    ServiceToolkit,  # Provides service tools
    ServiceResponse,  # Service response object
    ServiceExecStatus,  # Service execution status enumeration
)
import re


INSTRUCTION_PROMPT = """## What You Should Do:
1. First, analyze the current situation, and determine your goal.
2. Then, check if your goal is already achieved. If so, try to generate a response. Otherwise, think about how to achieve it with the help of provided tool functions.
3. Respond in the required format.

## Note:
1. Fully understand the tool functions and their arguments before using them.
2. You should decide if you need to use the tool functions, if not then return an empty list in "function" field.
3. Make sure the types and values of the arguments you provided to the tool functions are correct.
4. Don't take things for granted. For example, where you are, what's the time now, etc. You can try to use the tool functions to get information.
5. If the function execution fails, you should analyze the error and try to solve it.
"""  # noqa

# Example extract_predicted_pois function
import re
import json
# Improved extract_predicted_pois function
def extract_predicted_pois_combined(content):
    """
    Extract 'next_poi_id' list from response content and return parsed results.
    Consider multiple cases, including comments, non-numeric values, latitude/longitude, etc., and retain numeric IDs.
    
    Args:
        content: Response content to parse
        
    Returns:
        list: List of extracted POI IDs
    """
    # Remove comments (supports both // and /* */ formats)
    content_str = re.sub(r'//.*', '', content)
    content_str = re.sub(r'/\*.*?\*/', '', content_str, flags=re.DOTALL)

    poi_ids = []
    print(f"Backup parsing input content:{content}\n")

    # Try to parse content as JSON
    try:
        content_json = json.loads(content_str)
        next_poi_ids = content_json.get('next_poi_id', [])

        # Process each POI ID
        for poi_id in next_poi_ids:
            if isinstance(poi_id, str):
                # Extract numeric part from string
                match = re.search(r'\b(\d+)\b', poi_id)
                if match:
                    poi_ids.append(match.group(1))
            elif isinstance(poi_id, (int, float)):
                # Convert numeric types to string
                poi_ids.append(str(int(poi_id)))

    except (json.JSONDecodeError, TypeError):
        # If JSON parsing fails, try regex extraction
        
        # Try to extract from markdown code block
        code_block_pattern = r'```(?:json)?\s*({[^}]*"next_poi_id"\s*:\s*\[[^\]]*\][^}]*})\s*```'
        code_block_match = re.search(code_block_pattern, content, re.DOTALL)
        
        if code_block_match:
            # Extract content from code block
            code_block_content = code_block_match.group(1)
            
            # Try to parse as JSON
            try:
                code_block_json = json.loads(code_block_content)
                next_poi_ids = code_block_json.get('next_poi_id', [])
                
                # Process each POI ID
                for poi_id in next_poi_ids:
                    if isinstance(poi_id, str):
                        match = re.search(r'\b(\d+)\b', poi_id)
                        if match:
                            poi_ids.append(match.group(1))
                    elif isinstance(poi_id, (int, float)):
                        poi_ids.append(str(int(poi_id)))
            
            except (json.JSONDecodeError, TypeError):
                # If JSON parsing fails, try direct extraction
                next_poi_ids_pattern = r'"next_poi_id"\s*:\s*\[(.*?)\]'
                next_poi_ids_match = re.search(next_poi_ids_pattern, code_block_content, re.DOTALL)
                
                if next_poi_ids_match:
                    # Extract IDs from array
                    ids_str = next_poi_ids_match.group(1)
                    ids_list = re.split(r',\s*', ids_str)
                    
                    # Process each ID
                    for id_str in ids_list:
                        match = re.search(r'\b(\d+)\b', id_str)
                        if match:
                            poi_ids.append(match.group(1))
        
        else:
            # Try direct extraction from content
            next_poi_ids_pattern = r'"next_poi_id"\s*:\s*\[(.*?)\]'
            next_poi_ids_match = re.search(next_poi_ids_pattern, content, re.DOTALL)
            
            if next_poi_ids_match:
                # Extract IDs from array
                ids_str = next_poi_ids_match.group(1)
                ids_list = re.split(r',\s*', ids_str)
                
                # Process each ID
                for id_str in ids_list:
                    match = re.search(r'\b(\d+)\b', id_str)
                    if match:
                        poi_ids.append(match.group(1))
            
            else:
                # Last resort: extract all numbers
                numbers = re.findall(r'\b\d+\b', content)
                poi_ids = numbers

    return poi_ids


class CustomDictDialogAgent(DictDialogAgent):
    """
    Custom DictDialogAgent that extends the base DictDialogAgent with additional functionality.
    """
    def __init__(
        self,
        name: str,
        sys_prompt: str,
        model_config_name: str,
        parser: Optional[ParserBase] = None,
        memory_size: int = 0,
    ):
        super().__init__(
            name=name,
            sys_prompt=sys_prompt,
            model_config_name=model_config_name,
            parser=parser,
            memory_size=memory_size,
        )

    def reply(self, message: Msg) -> Msg:
        """
        Generate a reply to the given message.
        
        Args:
            message: Input message
            
        Returns:
            Msg: Response message
        """
        try:
            return super().reply(message)
        except ResponseParsingError as e:
            # Handle parsing error
            logger.warning(f"Response parsing error: {e}")
            
            # Extract content from error
            content = str(e)
            match = re.search(r"Failed to parse response: (.*)", content)
            if match:
                content = match.group(1)
            
            # Create response message
            response = Msg(
                name=self.name,
                content=content,
                role="assistant",
            )
            
            return response


class CustomReActAgent(ReActAgent):
    """
    Custom ReActAgent that extends the base ReActAgent with additional functionality.
    """
    def __init__(
        self,
        name: str,
        sys_prompt: str,
        model_config_name: str,
        service_toolkit: Optional[ServiceToolkit] = None,
        instruction_prompt: str = INSTRUCTION_PROMPT,
        memory_size: int = 0,
    ):
        super().__init__(
            name=name,
            sys_prompt=sys_prompt,
            model_config_name=model_config_name,
            service_toolkit=service_toolkit,
            instruction_prompt=instruction_prompt,
            memory_size=memory_size,
        )

    def reply(self, message: Msg) -> Msg:
        """
        Generate a reply to the given message.
        
        Args:
            message: Input message
            
        Returns:
            Msg: Response message
        """
        try:
            return super().reply(message)
        except ResponseParsingError as e:
            # Handle parsing error
            logger.warning(f"Response parsing error: {e}")
            
            # Extract content from error
            content = str(e)
            match = re.search(r"Failed to parse response: (.*)", content)
            if match:
                content = match.group(1)
            
            # Create response message
            response = Msg(
                name=self.name,
                content=content,
                role="assistant",
            )
            
            return response


class CustomDialogAgent(DialogAgent):
    """
    Custom DialogAgent that extends the base DialogAgent with additional functionality.
    """
    def __init__(
        self,
        name: str,
        sys_prompt: str,
        model_config_name: str,
        memory_size: int = 0,
    ):
        super().__init__(
            name=name,
            sys_prompt=sys_prompt,
            model_config_name=model_config_name,
            memory_size=memory_size,
        )

    def reply(self, message: Msg) -> Msg:
        """
        Generate a reply to the given message.
        
        Args:
            message: Input message
            
        Returns:
            Msg: Response message
        """
        try:
            return super().reply(message)
        except ResponseParsingError as e:
            # Handle parsing error
            logger.warning(f"Response parsing error: {e}")
            
            # Extract content from error
            content = str(e)
            match = re.search(r"Failed to parse response: (.*)", content)
            if match:
                content = match.group(1)
            
            # Create response message
            response = Msg(
                name=self.name,
                content=content,
                role="assistant",
            )
            
            return response
