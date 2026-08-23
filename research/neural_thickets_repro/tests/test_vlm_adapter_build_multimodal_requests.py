"""Tests for vlm_adapter.build_multimodal_requests() -- new file, since test_vlm_adapter.py
is entirely `pytest.importorskip("ray")`-gated at module scope and this function needs no
ray/vllm/GPU at all, only a real tokenizer-shaped fake + real PIL images.
"""
from types import SimpleNamespace

from neural_thickets_repro.vlm_adapter import build_multimodal_requests


def _fake_tokenizer():
    return SimpleNamespace(apply_chat_template=lambda messages, add_generation_prompt, tokenize: "TEXT:" + str(messages))


def test_image_actually_reaches_the_request_dict(tiny_image_factory):
    image = tiny_image_factory()
    messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    requests = build_multimodal_requests([(messages, image)], _fake_tokenizer())

    assert len(requests) == 1
    assert "multi_modal_data" in requests[0]
    assert requests[0]["multi_modal_data"]["image"] is image


def test_none_image_omits_multi_modal_data_key_entirely():
    messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    requests = build_multimodal_requests([(messages, None)], _fake_tokenizer())

    assert len(requests) == 1
    assert "multi_modal_data" not in requests[0]


def test_multiple_requests_preserve_order(tiny_image_factory):
    img_a = tiny_image_factory(color=(255, 0, 0))
    img_b = tiny_image_factory(color=(0, 255, 0))
    messages_a = [{"role": "user", "content": [{"type": "text", "text": "a"}]}]
    messages_b = [{"role": "user", "content": [{"type": "text", "text": "b"}]}]

    requests = build_multimodal_requests([(messages_a, img_a), (messages_b, None), (messages_b, img_b)], _fake_tokenizer())

    assert len(requests) == 3
    assert requests[0]["multi_modal_data"]["image"] is img_a
    assert "multi_modal_data" not in requests[1]
    assert requests[2]["multi_modal_data"]["image"] is img_b


def test_prompt_text_comes_from_tokenizer_chat_template(tiny_image_factory):
    image = tiny_image_factory()
    messages = [{"role": "user", "content": [{"type": "text", "text": "specific question"}]}]
    requests = build_multimodal_requests([(messages, image)], _fake_tokenizer())
    assert "specific question" in requests[0]["prompt"]
