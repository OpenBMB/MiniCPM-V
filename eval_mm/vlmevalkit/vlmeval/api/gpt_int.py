import json
import warnings
import requests
from ..smp import *
from .gpt import GPT_context_window, OpenAIWrapper

url = 'http://ecs.sv.us.alles-apin.openxlab.org.cn/v1/openai/v2/text/chat'
headers = {
    'Content-Type': 'application/json'
}


class OpenAIWrapperInternal(OpenAIWrapper):

    is_api: bool = True

    def __init__(self,
                 model: str = 'gpt-3.5-turbo-0613',
                 retry: int = 5,
                 wait: int = 3,
                 verbose: bool = True,
                 system_prompt: str = None,
                 temperature: float = 0,
                 timeout: int = 60,
                 max_tokens: int = 1024,
                 img_size: int = 512,
                 img_detail: str = 'low',
                 **kwargs):

        self.model = model
        if 'KEYS' in os.environ and osp.exists(os.environ['KEYS']):
            keys = load(os.environ['KEYS'])
            headers['alles-apin-token'] = keys.get('alles-apin-token', '')
        elif 'ALLES' in os.environ:
            headers['alles-apin-token'] = os.environ['ALLES']
        self.headers = headers
        self.temperature = temperature
        self.timeout = timeout
        self.max_tokens = max_tokens

        assert img_size > 0 or img_size == -1
        self.img_size = img_size
        assert img_detail in ['high', 'low']
        self.img_detail = img_detail

        super(OpenAIWrapper, self).__init__(
            wait=wait, retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

    def generate_inner(self, inputs, **kwargs) -> str:
        input_msgs = self.prepare_inputs(inputs)

        temperature = kwargs.pop('temperature', self.temperature)
        max_tokens = kwargs.pop('max_tokens', self.max_tokens)

        # Held out 100 tokens as buffer
        context_window = GPT_context_window(self.model)
        max_tokens = min(max_tokens, context_window - self.get_token_len(inputs))
        if 0 < max_tokens <= 100:
            print('Less than 100 tokens left, may exceed the context window with some additional meta symbols. ')
        if max_tokens <= 0:
            return 0, self.fail_msg + 'Input string longer than context window. ', 'Length Exceeded. '

        payload = dict(
            model=self.model,
            messages=input_msgs,
            max_tokens=max_tokens,
            n=1,
            stop=None,
            timeout=self.timeout,
            temperature=temperature,
            **kwargs)

        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=self.timeout * 1.1)
        ret_code = response.status_code
        ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code

        answer = self.fail_msg
        try:
            resp_struct = json.loads(response.text)
            assert resp_struct['msg'] == 'ok' and resp_struct['msgCode'] == '10000', resp_struct
            answer = resp_struct['data']['choices'][0]['message']['content'].strip()
        except:
            pass
        return ret_code, answer, response


class GPT4V_Internal(OpenAIWrapperInternal):

    def generate(self, message, dataset=None):
        return super(GPT4V_Internal, self).generate(message)
