# To define various training samples for testing with g4 chat template
# Ref: https://github.ibm.com/ai-models-architectures/granite-chat-template/blob/main/granite_4.0/granite_4_chat_template_test.ipynb
# Ref to Granite-4.0 tiny preview: https://huggingface.co/ibm-granite/granite-4.0-tiny-preview 

import json


# Sample 1: single_turn_wo_think_block
sample1 = {
    'messages': [
        {"role": "user", "content": "Who?"},
        {"role": "assistant", "content": "LLM", "tool_calls": None}
    ],
    'tools': [],
    'documents': []
}

# Sample 2: single turn w/ think block
sample2 = {
    'messages': [
        {"role": "user", "content": "Who?"},
        {"role": "assistant", "content": "<think>I am thinking</think> LLM", "tool_calls": None}
    ],
    'tools': [],
    'documents': []
}

# Sample 3: single turn w/ thought field (should be like sample2 after applying ct)
sample3 = {
    'messages': [
        {"role": "user", "content": "Who?"},
        {"role": "assistant", "content": " LLM", "thought": "I am thinking", "tool_calls": None}
    ],
    'tools': [],
    'documents': []
}


# Sample 4: multi-turn with multi system msg
# Stripping out ALL but the LAST <think></think> blocks to save tokens

sample4 = {
    'messages': [
    {"role": "system", "content": "1st sys.msg!"},
    {"role": "user", "content": "What?"},
    {"role": "assistant", "content": "<think> This MIDDLE thinking block will be REMOVED by g4ct </think> g4ct"},
    {"role": "system", "content": "2nd sys.msg!"},
    {"role": "user", "content": "Why?"},
    {"role": "assistant", "content": "<think> This MIDDLE thinking block will ALSO be REMOVED by g4ct</think> due to g4ct"},
    {"role": "user", "content": "How abt this last one?"},
    {"role": "assistant", "content": "<think> This LAST thinking block is KEPT by g4ct!</think> due to g4ct"},
    ],
    'tools': [],
    'documents': []
}

# Multi-turn Tool Calling with Thinking
## 1. ALL <think></think> blocks are NOT stripped out. 
## 2. The tool calls are presented in <tool_call></tool_call> tags in assistant 
## 3. tool's content: treated as USER for MASKING (i.e. w/o computing loss for it)

sample5 = {
    'messages': [
        {"role": "user", "content": "*** USER1: What's the weather in Bengaluru?"},

        {"role": "assistant",
        "content": "<think> MODEL THINKING... </think>*** ASST2: get weather",
        "tool_calls": [
            {
                "function": {
                    "name": "get_weather",
                    "arguments": {"city": "Bengaluru"}
                }
            }
        ]
        },

        {"role": "tool", "content": "*** TOOL3 (treated as USER for MASKING): Bengaluru is sunny."},
        {"role": "tool", "content": "*** TOOL4 (treated as USER for MASKING): Humidity is around 40%."},

        {"role": "assistant",
        "content": "*** ASST5: get_weather",
        "thought": "User may want weather for both Bengaluru and Kolkata.",
        "tool_calls": [
            {
                "function": {
                    "name": "get_weather",
                    "arguments": {"city": "Bengaluru"}
                }
            },
            {
                "function": {
                    "name": "get_weather",
                    "arguments": {"city": "Kolkata"}
                }
            }
        ]
        },

        {"role": "tool", "content": "*** TOOL6 (treated as USER for MASKING): Bengaluru is still sunny."},
        {"role": "tool", "content": "*** TOOL7 (treated as USER for MASKING): Kolkata is cloudy."},

        {"role": "assistant",
        "content": "<think> Bengaluru is warmer, while Kolkata is cooler with rain. </think>*** ASST8: The weather in Bengaluru is sunny. Kolkata is rainy."}
    ],
    'tools': [
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
        }    
    ],
    'documents': []
}




# Document Based RAG
sample6 = {
    'messages': [
    {"role": "user", "content": "Who?"},
    {"role": "assistant", "content": "<think> 1st THINKING WILL BE REMOVED </think> LLM1"},
    {"role": "user", "content": "What?"},
    {"role": "assistant", "content": "<think> 2nd THINKING WILL BE KEPT </think> LLM2"},
    ],
    'tools': [],
    'documents': [
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
}

# With Tools + Documents + System Prompts
### tools are first presented followed by documents. 
### If multiple system prompts are present then the first system prompt is placed before tools and documents
sample7 = {
        'messages': [
        {"role": "user", "content": "*** USER1: What's the weather in Bengaluru?"},

        {"role": "assistant",
        "content": "<think> MODEL THINKING... </think>*** ASST2: get weather",
        "tool_calls": [
            {
                "function": {
                    "name": "get_weather",
                    "arguments": {"city": "Bengaluru"}
                }
            }
        ]
        },

        {"role": "tool", "content": "*** TOOL3 (treated as USER for MASKING): Bengaluru is sunny."},

        {"role": "tool", "content": "*** TOOL4 (treated as USER for MASKING): Humidity is around 40%."},

        {"role": "assistant",
        "content": "*** ASST5: get_weather",
        "thought": "User may want weather for both Bengaluru and Kolkata.",
        "tool_calls": [
            {
                "function": {
                    "name": "get_weather",
                    "arguments": {"city": "Bengaluru"}
                }
            },
            {
                "function": {
                    "name": "get_weather",
                    "arguments": {"city": "Kolkata"}
                }
            }
        ]
        },

        {"role": "tool", "content": "*** TOOL6 (treated as USER for MASKING): Bengaluru is still sunny."},
        {"role": "tool", "content": "*** TOOL7 (treated as USER for MASKING): Kolkata is cloudy."},

        {"role": "assistant",
        "content": "<think> Bengaluru is warmer, while Kolkata is cooler with rain. </think>*** ASST8: The weather in Bengaluru is sunny. Kolkata is rainy."}
    ],
    'tools':[
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
        }
    ],
    'documents': [
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
}



g4ct_dataset = {
    'messages': [
        sample1['messages'],
        sample2['messages'],
        sample3['messages'],
        sample4['messages'],
        sample5['messages'],
        sample6['messages'],
        sample7['messages']
    ],
    'tools': [
        sample1['tools'],
        sample2['tools'],
        sample3['tools'],
        sample4['tools'],
        sample5['tools'],
        sample6['tools'],
        sample7['tools']
    ],
    'documents': [
        sample1['documents'],
        sample2['documents'],
        sample3['documents'],
        sample4['documents'],
        sample5['documents'],
        sample6['documents'],
        sample7['documents']
    ]
}

def main():
    for i in range(len(g4ct_dataset['messages'])):
        sample = {
            'messages': g4ct_dataset['messages'][i],
            'tools': g4ct_dataset['tools'][i],
            'documents': g4ct_dataset['documents'][i]
        }
        print(f"\n=== Sample {i + 1} ===")
        print(json.dumps(sample, indent=2))
        


if __name__ == "__main__":
    # conda activate /proj/data-eng/replaybuffer/07-conda-env/open-instruct-g4l-env
    # cd /proj/data-eng/xhd/commonscript/fms-hf-tuning/src/02-chat-template/ && python ct-g4l-samples.py
    main()