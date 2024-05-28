import os
from vlmeval.api import OpenAIWrapper, OpenAIWrapperInternal
from vlmeval.smp import load_env

INTERNAL = os.environ.get('INTERNAL', 0)


def build_judge(**kwargs):
    model = kwargs.pop('model', None)
    load_env()
    LOCAL_LLM = os.environ.get('LOCAL_LLM', None)
    if LOCAL_LLM is None:
        model_map = {
            'gpt-4-turbo': 'gpt-4-1106-preview',
            'gpt-4-0613': 'gpt-4-0613',
            'gpt-4-0314': 'gpt-4-0314',
            'gpt-4-0125': 'gpt-4-0125-preview',
            'chatgpt-1106': 'gpt-3.5-turbo-1106',
            'chatgpt-0613': 'gpt-3.5-turbo-0613',
            'chatgpt-0125': 'gpt-3.5-turbo-0125'
        }
        model_version = model_map[model]
    else:
        model_version = LOCAL_LLM
    if INTERNAL:
        model = OpenAIWrapperInternal(model_version, **kwargs)
    else:
        model = OpenAIWrapper(model_version, **kwargs)
    return model
