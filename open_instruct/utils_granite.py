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
    messages = [
    {"role": "system", "content": "1st sys.msg!"},
    {"role": "user", "content": "What?"},
    {"role": "assistant", "content": "<think> This MIDDLE thinking block will be REMOVED by g4ct </think> g4ct"},
    {"role": "system", "content": "2nd sys.msg!"},
    {"role": "user", "content": "Why?"},
    {"role": "assistant", "content": "<think> This MIDDLE thinking block will ALSO be REMOVED by g4ct</think> due to g4ct"},
    {"role": "user", "content": "How abt this last one?"},
    {"role": "assistant", "content": "<think> This LAST thinking block is KEPT by g4ct!</think> due to g4ct"},
    ]
    return messages

def _get_default_tools():
    # used for testing granite4 chat template
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "GET WEATHER.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "NAME OF CITY"
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
                "description": "GET TIME.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "LOC.COORDINATES"
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
            "title": "Doc1 Title",
            "text": "Doc1 content.",
            "source": "Doc1 source"
        },
        {
            "doc_id": 2,
            "title": "Doc2 title",
            "text": "Doc2 content.",
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


def stop_debugging(accelerator: Accelerator = None,msg:str=None) -> None:
    """
    Stops debugging and cleans up distributed resources.
    Args:
        accelerator (Accelerator): The accelerator instance managing distributed training.
    """
    
    # ensure all processes wait until the main process finishes 
    
    if msg is not None:
        if accelerator is not None and accelerator.is_local_main_process:
            accelerator.print(f"\n\n** {msg} **\n")
        elif accelerator is None:
            print(f"\n\n** {msg} **\n")
            
    if accelerator is not None:
        accelerator.wait_for_everyone()
        accelerator.end_training()
        accelerator.free_memory()
    sys.exit("== STOP DEBUGGING ==")      