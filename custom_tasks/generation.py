import torch

from cache import CompressedCache
from utils.model import get_device


@torch.no_grad()
def generate(model, tokenizer, cache_config, prompts, max_new_tokens):
    device = get_device(model)
    width = max(len(prompt) for prompt in prompts)
    ids = torch.full(
        (len(prompts), width),
        tokenizer.pad_token_id,
        dtype=torch.long,
        device=device,
    )
    mask = torch.zeros_like(ids)
    for row, prompt in enumerate(prompts):
        ids[row, width - len(prompt) :] = torch.as_tensor(prompt, device=device)
        mask[row, width - len(prompt) :] = 1

    cache = CompressedCache(
        config=cache_config, cache_context={"padding_mask": mask}, verbose=False
    )
    storage = {}

    def capture(module, inputs, output):
        if not storage:
            storage.update(cache.prefill_compression_stats(mask))

    handle = model.register_forward_hook(capture)
    try:
        output = model.generate(
            input_ids=ids,
            attention_mask=mask,
            past_key_values=cache,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            logits_to_keep=1,
            pad_token_id=tokenizer.pad_token_id,
        )
    finally:
        handle.remove()
    texts = tokenizer.batch_decode(output[:, width:], skip_special_tokens=True)
    return [text.strip() for text in texts], storage
