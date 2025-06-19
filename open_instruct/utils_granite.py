from transformers import PreTrainedTokenizer
from typing import List, Dict, Optional, Union
from accelerate import Accelerator
import sys


def _get_simple_messages():
    # used for testing granite4 chat template
    messages = [
            {"role": "user", "content": "Who?"},
            {"role": "assistant", "content": "LLM"},
        ]
    return messages

def _get_default_messages():
    # used for testing granite4 chat template
    # messages = [
    #         {"role": "user", "content": "Who?"},
    #         {"role": "assistant", "content": "LLM"},
    #     ]
    messages = [
        {"role": "system", "content": "You are a weather assistant that responds with relevant function calls instead of natural language."},
        {"role": "user", "content": "What's the weather like in Bengaluru?"},
        {"role": "assistant", "content": "get_coordinates(city='Bengaluru')"},
        {"role": "system", "content": "Coordinates retrieved successfully. You can now use weather-related functions with latitude and longitude."},
        {"role": "user", "content": "Can you tell me the current temperature there?"},
        {"role": "assistant", "content": "get_current_weather(lat=12.97, lon=77.59)"},
        {"role": "system", "content": "User has requested a multi-day forecast. Switch to forecast mode."},
        {"role": "user", "content": "Actually, I need the 3-day forecast for planning a trip."},
        {"role": "assistant", "content": "get_weather_forecast(lat=12.97, lon=77.59, days=3)"},
    ]
    return messages

def _get_default_tools():
    # used for testing granite4 chat template
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather for a specified city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "Name of the city"
                        }
                    },
                    "required": ["city"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_time",
                "description": "Get the current time for a specified location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "Coordinates of the location"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    return tools

def _get_default_RAG_documents():
    # used for testing granite4 chat template
    documents = [
        {
            "doc_id": 1,
            "title": "",
            "text": "From the early 12th century, French builders developed the Gothic style, marked by the use of rib vaults, pointed arches, flying buttresses, and large stained glass windows. It was used mainly in churches and cathedrals, and continued in use until the 16th century in much of Europe. Classic examples of Gothic architecture include Chartres Cathedral and Reims Cathedral in France as well as Salisbury Cathedral in England. Stained glass became a crucial element in the design of churches, which continued to use extensive wall-paintings, now almost all lost.",
            "source": ""
        },
        {
            "doc_id": 2,
            "title": "",
            "text": "From long time ago, French builders developed the Gothic style, marked by the use of rib vaults, pointed arches, flying buttresses, and large stained glass windows. It was used mainly in churches and cathedrals, and continued in use until the 16th century in much of Europe. Classic examples of Gothic architecture include Chartres Cathedral and Reims Cathedral in France as well as Salisbury Cathedral in England. Stained glass became a crucial element in the design of churches, which continued to use extensive wall-paintings, now almost all lost.",
            "source": ""
        },
        {
            "doc_id": 3,
            "title": "",
            "text": "From yesterday, French builders developed the Gothic style, marked by the use of rib vaults, pointed arches, flying buttresses, and large stained glass windows. It was used mainly in churches and cathedrals, and continued in use until the 16th century in much of Europe. Classic examples of Gothic architecture include Chartres Cathedral and Reims Cathedral in France as well as Salisbury Cathedral in England. Stained glass became a crucial element in the design of churches, which continued to use extensive wall-paintings, now almost all lost.",
            "source": ""
        }
    ]
    return documents
    
def add_special_chat_tokens(tokenizer, add_special_tokens:list):
    existing_special_tokens = tokenizer.special_tokens_map.get("additional_special_tokens", [])
    new_special_tokens = [t for t in add_special_tokens if t not in existing_special_tokens]
    if new_special_tokens:  
        all_special_tokens = existing_special_tokens + new_special_tokens
        tokenizer.add_special_tokens({"additional_special_tokens": all_special_tokens})
            
    return tokenizer
    


def debug_chat_template_tokenization(
    tokenizer: PreTrainedTokenizer,
    messages: Union[None, str, List[Dict[str, str]]] = None, # or "Default" or a list [...]
    tools: Union[None, str, List[Dict[str, str]]] = None, # or "Default" or a list [...]
    documents: Union[None, str, List[Dict[str, str]]] = None, # or "Default" or a list [...]
) -> None:
    """
    Applies chat template to a sample of messages/tools/documents and tokenizes the resulting text.

    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer instance.
        messages: Either "Default", None, or a list of {"role": ..., "content": ...} dicts.
        tools: Either "Default", None, or a list of tool dicts.
        documents: Either "Default", None, or a list of document dicts.
    Example:
        debug_chat_template_tokenization(tokenizer)
        debug_chat_template_tokenization(tokenizer,"Default","Default","Default")

    """
    print(
        f"\n== Tokenizer info: {len(tokenizer):,} tokens (vocab_size={tokenizer.vocab_size:,}) =="
        f"\n== Special Tokens Map (len={len(tokenizer.special_tokens_map)}):"
    )
    
    # special_token -> tokenID:
    for name, token_str in tokenizer.special_tokens_map.items():
        token_id = tokenizer.convert_tokens_to_ids(token_str)
        print(f"  {name:>20}: '{token_str}' --> ID: {token_id}")
    
    # default messages:    
    if messages in ("Default", None):
        messages = _get_simple_messages() if messages is None else _get_default_messages()
        print(f"\n== Messages:\n{messages}")
    
    # Collect optional inputs for applying chat template:
    additional_inputs = {}
    if tools is not None:
        additional_inputs["tools"] = _get_default_tools() if tools == "Default" else tools
        print(f"\n== Tools:\n{tools}")
    
    if documents is not None:
        additional_inputs["documents"] = _get_default_RAG_documents() if documents == "Default" else documents
        print(f"\n== Documents:\n{documents}")
    
    text = tokenizer.apply_chat_template(messages, 
                                         tokenize=False,
                                         **additional_inputs,
                                         )    

    tokens = tokenizer.encode(text, add_special_tokens=False)
    decoded_tokens = [tokenizer.decode([t]) for t in tokens]
    zipped = list(zip(tokens, decoded_tokens))
    
    
    print(f"\n== Chat Template Output:\n{text}")
    print(f"\n== Tokenization Output (len={len(tokens)}):\n{tokens}\n")
    for token_id, token_str in zipped:
        print(f"{token_id:6d} -> `{token_str}`")


def stop_debugging(accelerator: Accelerator) -> None:
    """
    Stops debugging and cleans up distributed resources.
    Args:
        accelerator (Accelerator): The accelerator instance managing distributed training.
    """
    
    # ensure all processes wait until the main process finishes 
    accelerator.wait_for_everyone()
    
    # clean up distributed resources
    accelerator.end_training()
    accelerator.free_memory()
    sys.exit("== STOP DEBUGGING ==")      