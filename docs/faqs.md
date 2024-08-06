### FAQs

<details>
<summary>Q:  How to choose between sampling or beam search for inference </summary>

In various scenarios, the quality of results obtained from beam search and sampling decoding strategies can vary. You can determine your decoding strategy based on the following aspects:

If you have the following needs, consider using sampling decoding:

1. You require faster inference speed.
2. You wish for a streaming generation approach.
3. Your task necessitates some open-ended responses.

If your task is about providing deterministic answers, you might want to experiment with beam search to see if it can achieve better outcomes.
</details>


<details>
<summary>Q: How to ensure that the model generates results of sufficient length</summary>

We've observed that during multi-language inference on MiniCPM-V 2.6, the generation sometimes ends prematurely. You can improve the results by passing a `min_new_tokens` parameter.
```python
res = model.chat(
    image=None,
    msgs=msgs,
    tokenizer=tokenizer,
    min_new_tokens=100
)
```
</details>
