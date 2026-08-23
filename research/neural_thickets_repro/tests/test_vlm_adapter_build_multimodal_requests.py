"""Tests for vlm_adapter.build_multimodal_requests() -- new file, since test_vlm_adapter.py
is entirely `pytest.importorskip("ray")`-gated at module scope and this function needs no
ray/vllm/GPU at all, only a real tokenizer-shaped fake + real PIL images.
"""
import pytest
from types import SimpleNamespace

from neural_thickets_repro.vlm_adapter import (
    IMAGE_PLACEHOLDER_TOKEN,
    TextOnlyRequestInvariantError,
    _strip_image_content_items,
    build_multimodal_requests,
)


def _fake_tokenizer():
    return SimpleNamespace(apply_chat_template=lambda messages, add_generation_prompt, tokenize: "TEXT:" + str(messages))


def _qwen_style_chat_template(messages, add_generation_prompt, tokenize):
    """A minimal fake reproducing the REAL bug: any {"type": "image"} content item renders
    into the literal image placeholder token, regardless of whether an actual image is
    attached via multi_modal_data -- this is what Qwen2.5-VL's real chat template does.
    """
    rendered = []
    for message in messages:
        for item in message.get("content", []):
            if isinstance(item, dict) and item.get("type") == "image":
                rendered.append(f"<|vision_start|>{IMAGE_PLACEHOLDER_TOKEN}<|vision_end|>")
            elif isinstance(item, dict) and item.get("type") == "text":
                rendered.append(item["text"])
    return " ".join(rendered)


def _qwen_style_tokenizer():
    return SimpleNamespace(apply_chat_template=_qwen_style_chat_template)


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


# ---------------------------------------------------------------------------------------
# _strip_image_content_items -- the core text-only crash fix
# ---------------------------------------------------------------------------------------

def test_strip_image_content_items_removes_image_blocks_keeps_text():
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "hi"}]}]
    stripped = _strip_image_content_items(messages)
    assert stripped == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]


def test_strip_image_content_items_does_not_mutate_the_original():
    original_content = [{"type": "image"}, {"type": "text", "text": "hi"}]
    messages = [{"role": "user", "content": original_content}]
    _strip_image_content_items(messages)
    assert original_content == [{"type": "image"}, {"type": "text", "text": "hi"}]  # untouched


def test_strip_image_content_items_handles_multiple_messages_and_multiple_image_blocks():
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": "a"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "b"}]},
    ]
    stripped = _strip_image_content_items(messages)
    assert stripped[0]["content"] == [{"type": "text", "text": "a"}]
    assert stripped[1]["content"] == [{"type": "text", "text": "b"}]


def test_strip_image_content_items_leaves_non_list_content_alone():
    messages = [{"role": "user", "content": "plain string content"}]
    assert _strip_image_content_items(messages) == messages


# ---------------------------------------------------------------------------------------
# Text-only real-bug reproduction: a Qwen-style chat template that renders an image
# placeholder token for any {"type": "image"} content item regardless of multi_modal_data.
# ---------------------------------------------------------------------------------------

def test_text_only_request_never_contains_the_image_placeholder_token():
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "how many objects?"}]}]
    requests = build_multimodal_requests([(messages, None)], _qwen_style_tokenizer())

    assert IMAGE_PLACEHOLDER_TOKEN not in requests[0]["prompt"]
    assert "multi_modal_data" not in requests[0]
    assert "how many objects?" in requests[0]["prompt"]


def test_text_only_request_text_is_otherwise_unchanged(tiny_image_factory):
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "what color is the chair?"}]}]

    text_only_requests = build_multimodal_requests([(messages, None)], _qwen_style_tokenizer())
    with_image_requests = build_multimodal_requests([(messages, tiny_image_factory())], _qwen_style_tokenizer())

    # text-only: the image content item was stripped before rendering, so only the question
    # text is rendered -- no placeholder token anywhere in it.
    assert text_only_requests[0]["prompt"] == "what color is the chair?"
    # with-image: the SAME question text, plus the (legitimate, image-is-attached) placeholder.
    assert with_image_requests[0]["prompt"] == f"<|vision_start|>{IMAGE_PLACEHOLDER_TOKEN}<|vision_end|> what color is the chair?"


def test_correct_image_condition_unaffected_by_the_stripping_logic(tiny_image_factory):
    image = tiny_image_factory()
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "q"}]}]
    requests = build_multimodal_requests([(messages, image)], _qwen_style_tokenizer())

    assert IMAGE_PLACEHOLDER_TOKEN in requests[0]["prompt"]  # image IS attached -- placeholder legitimately present
    assert requests[0]["multi_modal_data"]["image"] is image


def test_shuffled_image_condition_unaffected_by_the_stripping_logic(tiny_image_factory):
    """The image-sanity "shuffled" condition still attaches a real (just wrong) image --
    must behave identically to the correct-image path, never treated as text-only.
    """
    wrong_image = tiny_image_factory(color=(9, 9, 9))
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "q"}]}]
    requests = build_multimodal_requests([(messages, wrong_image)], _qwen_style_tokenizer())

    assert IMAGE_PLACEHOLDER_TOKEN in requests[0]["prompt"]
    assert requests[0]["multi_modal_data"]["image"] is wrong_image


def test_invariant_check_raises_if_placeholder_survives_despite_no_image(tiny_image_factory):
    """Directly exercises the pre-generation invariant/check: even if some future adapter
    change reintroduces an image placeholder into a text-only request's rendered prompt, this
    must hard-fail BEFORE reaching vLLM, not silently send a request that would crash it.
    """
    tokenizer = SimpleNamespace(apply_chat_template=lambda messages, add_generation_prompt, tokenize: f"prefix {IMAGE_PLACEHOLDER_TOKEN} suffix")
    messages = [{"role": "user", "content": [{"type": "text", "text": "q"}]}]

    with pytest.raises(TextOnlyRequestInvariantError, match="image_pad"):
        build_multimodal_requests([(messages, None)], tokenizer)


def test_multiple_requests_mixed_conditions_all_correct(tiny_image_factory):
    """End-to-end sanity across all three image-sanity conditions in one batch, using the
    real Qwen-style chat-template fake.
    """
    correct_image = tiny_image_factory(color=(1, 1, 1))
    shuffled_image = tiny_image_factory(color=(2, 2, 2))
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "q"}]}]

    requests = build_multimodal_requests(
        [(messages, correct_image), (messages, shuffled_image), (messages, None)], _qwen_style_tokenizer(),
    )

    assert requests[0]["multi_modal_data"]["image"] is correct_image
    assert IMAGE_PLACEHOLDER_TOKEN in requests[0]["prompt"]
    assert requests[1]["multi_modal_data"]["image"] is shuffled_image
    assert IMAGE_PLACEHOLDER_TOKEN in requests[1]["prompt"]
    assert "multi_modal_data" not in requests[2]
    assert IMAGE_PLACEHOLDER_TOKEN not in requests[2]["prompt"]
