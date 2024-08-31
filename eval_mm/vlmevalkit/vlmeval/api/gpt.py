from ..smp import *
import os
import sys
from .base import BaseAPI

APIBASES = {
    'OFFICIAL': 'https://api.openai.com/v1/chat/completions',
}


def GPT_context_window(model):
    length_map = {
        'gpt-4': 8192,
        'gpt-4-0613': 8192,
        'gpt-4-turbo-preview': 128000,
        'gpt-4-1106-preview': 128000,
        'gpt-4-0125-preview': 128000,
        'gpt-4-vision-preview': 128000,
        'gpt-4-turbo': 128000,
        'gpt-4-turbo-2024-04-09': 128000,
        'gpt-3.5-turbo': 16385,
        'gpt-3.5-turbo-0125': 16385,
        'gpt-3.5-turbo-1106': 16385,
        'gpt-3.5-turbo-instruct': 4096,
    }
    if model in length_map:
        return length_map[model]
    else:
        return 128000


class OpenAIWrapper(BaseAPI):

    is_api: bool = True

    def __init__(self,
                 model: str = 'gpt-3.5-turbo-0613',
                 retry: int = 5,
                 wait: int = 5,
                 key: str = None,
                 verbose: bool = True,
                 system_prompt: str = None,
                 temperature: float = 0,
                 timeout: int = 60,
                 api_base: str = None,
                 max_tokens: int = 1024,
                 img_size: int = 512,
                 img_detail: str = 'low',
                 use_azure: bool = False,
                 **kwargs):

        self.model = model
        self.cur_idx = 0
        self.fail_msg = 'Failed to obtain answer via API. '
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.use_azure = use_azure

        if 'step-1v' in model:
            env_key = os.environ.get('STEPAI_API_KEY', '')
            if key is None:
                key = env_key
        elif 'yi-vision' in model:
            env_key = os.environ.get('YI_API_KEY', '')
            if key is None:
                key = env_key
        else:
            if use_azure:
                env_key = os.environ.get('AZURE_OPENAI_API_KEY', None)
                assert env_key is not None, 'Please set the environment variable AZURE_OPENAI_API_KEY. '

                if key is None:
                    key = env_key
                assert isinstance(key, str), (
                    'Please set the environment variable AZURE_OPENAI_API_KEY to your openai key. '
                )
            else:
                env_key = os.environ.get('OPENAI_API_KEY', '')
                if key is None:
                    key = env_key
                assert isinstance(key, str) and key.startswith('sk-'), (
                    f'Illegal openai_key {key}. '
                    'Please set the environment variable OPENAI_API_KEY to your openai key. '
                )

        self.key = key
        assert img_size > 0 or img_size == -1
        self.img_size = img_size
        assert img_detail in ['high', 'low']
        self.img_detail = img_detail
        self.timeout = timeout

        super().__init__(wait=wait, retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

        if use_azure:
            api_base_template = (
                '{endpoint}openai/deployments/{deployment_name}/chat/completions?api-version={api_version}'
            )
            endpoint = os.getenv('AZURE_OPENAI_ENDPOINT', None)
            assert endpoint is not None, 'Please set the environment variable AZURE_OPENAI_ENDPOINT. '
            deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', None)
            assert deployment_name is not None, 'Please set the environment variable AZURE_OPENAI_DEPLOYMENT_NAME. '
            api_version = os.getenv('OPENAI_API_VERSION', None)
            assert api_version is not None, 'Please set the environment variable OPENAI_API_VERSION. '

            self.api_base = api_base_template.format(
                endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
                deployment_name=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
                api_version=os.getenv('OPENAI_API_VERSION')
            )
        else:
            if api_base is None:
                if 'OPENAI_API_BASE' in os.environ and os.environ['OPENAI_API_BASE'] != '':
                    self.logger.info('Environment variable OPENAI_API_BASE is set. Will use it as api_base. ')
                    api_base = os.environ['OPENAI_API_BASE']
                else:
                    api_base = 'OFFICIAL'

            assert api_base is not None

            if api_base in APIBASES:
                self.api_base = APIBASES[api_base]
            elif api_base.startswith('http'):
                self.api_base = api_base
            else:
                self.logger.error('Unknown API Base. ')
                sys.exit(-1)

        self.logger.info(f'Using API Base: {self.api_base}; API Key: {self.key}')

    # inputs can be a lvl-2 nested list: [content1, content2, content3, ...]
    # content can be a string or a list of image & text
    def prepare_itlist(self, inputs):
        assert np.all([isinstance(x, dict) for x in inputs])
        has_images = np.sum([x['type'] == 'image' for x in inputs])
        if has_images:
            content_list = []
            for msg in inputs:
                if msg['type'] == 'text':
                    content_list.append(dict(type='text', text=msg['value']))
                elif msg['type'] == 'image':
                    from PIL import Image
                    img = Image.open(msg['value'])
                    b64 = encode_image_to_base64(img, target_size=self.img_size)
                    img_struct = dict(url=f'data:image/jpeg;base64,{b64}', detail=self.img_detail)
                    content_list.append(dict(type='image_url', image_url=img_struct))
        else:
            assert all([x['type'] == 'text' for x in inputs])
            text = '\n'.join([x['value'] for x in inputs])
            content_list = [dict(type='text', text=text)]
        return content_list

    def prepare_inputs(self, inputs):
        input_msgs = []
        if self.system_prompt is not None:
            input_msgs.append(dict(role='system', content=self.system_prompt))
        assert isinstance(inputs, list) and isinstance(inputs[0], dict)
        assert np.all(['type' in x for x in inputs]) or np.all(['role' in x for x in inputs]), inputs
        if 'role' in inputs[0]:
            assert inputs[-1]['role'] == 'user', inputs[-1]
            for item in inputs:
                input_msgs.append(dict(role=item['role'], content=self.prepare_itlist(item['content'])))
        else:
            input_msgs.append(dict(role='user', content=self.prepare_itlist(inputs)))
        return input_msgs

    def generate_inner(self, inputs, **kwargs) -> str:
        input_msgs = self.prepare_inputs(inputs)
        temperature = kwargs.pop('temperature', self.temperature)
        max_tokens = kwargs.pop('max_tokens', self.max_tokens)

        context_window = GPT_context_window(self.model)
        max_tokens = min(max_tokens, context_window - self.get_token_len(inputs))
        if 0 < max_tokens <= 100:
            self.logger.warning(
                'Less than 100 tokens left, '
                'may exceed the context window with some additional meta symbols. '
            )
        if max_tokens <= 0:
            return 0, self.fail_msg + 'Input string longer than context window. ', 'Length Exceeded. '

        # Will send request if use Azure, dk how to use openai client for it
        if self.use_azure:
            headers = {'Content-Type': 'application/json', 'api-key': self.key}
        else:
            headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.key}'}
        payload = dict(
            model=self.model,
            messages=input_msgs,
            max_tokens=max_tokens,
            n=1,
            temperature=temperature,
            **kwargs)
        response = requests.post(
            self.api_base,
            headers=headers, data=json.dumps(payload), timeout=self.timeout * 1.1)
        ret_code = response.status_code
        ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code
        answer = self.fail_msg
        try:
            resp_struct = json.loads(response.text)
            answer = resp_struct['choices'][0]['message']['content'].strip()
        except:
            pass
        return ret_code, answer, response

    def get_image_token_len(self, img_path, detail='low'):
        import math
        if detail == 'low':
            return 85

        im = Image.open(img_path)
        height, width = im.size
        if width > 1024 or height > 1024:
            if width > height:
                height = int(height * 1024 / width)
                width = 1024
            else:
                width = int(width * 1024 / height)
                height = 1024

        h = math.ceil(height / 512)
        w = math.ceil(width / 512)
        total = 85 + 170 * h * w
        return total

    def get_token_len(self, inputs) -> int:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model(self.model)
        except:
            enc = tiktoken.encoding_for_model('gpt-4')
        assert isinstance(inputs, list)
        tot = 0
        for item in inputs:
            if 'role' in item:
                tot += self.get_token_len(item['content'])
            elif item['type'] == 'text':
                tot += len(enc.encode(item['value']))
            elif item['type'] == 'image':
                tot += self.get_image_token_len(item['value'], detail=self.img_detail)
        return tot


class GPT4V(OpenAIWrapper):

    def generate(self, message, dataset=None):
        return super(GPT4V, self).generate(message)
