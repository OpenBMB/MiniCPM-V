## about xinference
xinference is a unified inference platform that provides a unified interface for different inference engines. It supports LLM, text generation, image generation, and more.but it's not bigger than swift too much.

## xinference install
```shell
pip install "xinference[all]"
```

## quick start
1. start xinference
```shell
xinference
```
2. start the web ui.
3. Search for "MiniCPM-Llama3-V-2_5" in the search box.
[alt text](../assets/xinferenc_demo_image/xinference_search_box.png)
4. find and click the MiniCPM-Llama3-V-2_5 button.
5. follow the config and launch the model.
```plaintext
Model engine : Transformers
model format : pytorch
Model size   : 8
quantization : none
N-GPU        : auto
Replica      : 1
```
6. after first click the launch button,xinference will download the model from huggingface. we should click the webui button.
![alt text](../assets/xinferenc_demo_image/xinference_webui_button.png)
7. upload the image and chatting with the MiniCPM-Llama3-V-2_5

## local MiniCPM-Llama3-V-2_5 launch
1. start xinference
```shell
xinference
```
2. start the web ui.
3. To register a new model, follow these steps: the settings highlighted in red are fixed and cannot be changed, whereas others are customizable according to your needs. Complete the process by clicking the 'Register Model' button.
![alt text](../assets/xinferenc_demo_image/xinference_register_model1.png)
![alt text](../assets/xinferenc_demo_image/xinference_register_model2.png)
4. After completing the model registration, proceed to 'Custom Models' and locate the model you just registered.
5. follow the config and launch the model.
```plaintext
Model engine : Transformers
model format : pytorch
Model size   : 8
quantization : none
N-GPU        : auto
Replica      : 1
```
6. after first click the launch button,xinference will download the model from huggingface. we should click the chat button.
![alt text](../assets/xinferenc_demo_image/xinference_webui_button.png)
7. upload the image and chatting with the MiniCPM-Llama3-V-2_5

## FAQ
1. Why can't the sixth step open the WebUI?
maybe your firewall or mac os to prevent the web to open.