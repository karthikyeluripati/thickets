"""Regression test for the known-good runtime pins in requirements-gpu.txt (see that file's
own comment for the full story: an unpinned vllm==0.11.0 install pulled in transformers 5.15.1
by default on a real RunPod, breaking vllm+Qwen2.5-VL with an mrope/rope_type conflict).
Parses the requirements files directly -- does not install or import anything.
"""
import re
from pathlib import Path

REQUIREMENTS_DIR = Path(__file__).resolve().parents[1] / "requirements"


def _parse_requirements(path: Path) -> dict:
    """Returns {package_name: full_requirement_line} for simple `name==version` /
    `name>=version` / bare `name` lines -- good enough for this file's own simple format,
    not a general requirements-file parser.
    """
    parsed = {}
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-r "):
            continue
        match = re.match(r"^([A-Za-z0-9_.\[\]-]+)", line)
        if match:
            parsed[match.group(1).lower()] = line
    return parsed


def test_vllm_transformers_tokenizers_are_pinned_to_the_confirmed_compatible_combination():
    reqs = _parse_requirements(REQUIREMENTS_DIR / "requirements-gpu.txt")

    assert reqs["vllm"] == "vllm==0.11.0", reqs.get("vllm")
    assert reqs["transformers"] == "transformers==4.57.1", reqs.get("transformers")
    assert reqs["tokenizers"] == "tokenizers==0.22.1", reqs.get("tokenizers")
    assert reqs["datasets"] == "datasets==5.0.1", reqs.get("datasets")


def test_hf_transfer_is_present_for_hf_hub_enable_hf_transfer():
    reqs = _parse_requirements(REQUIREMENTS_DIR / "requirements-gpu.txt")
    assert "hf_transfer" in reqs


def test_torch_is_not_pinned_to_an_exact_build_only_a_floor():
    """Preserve torch>=2.6.0 (or a tighter floor if ever justified) -- never an exact
    torch==X.Y.Z(+cuNNN) pin, which would force an unnecessary/possibly-incompatible
    reinstall on a RunPod template that already ships a CUDA-matched torch build.
    """
    cpu_reqs = _parse_requirements(REQUIREMENTS_DIR / "requirements-cpu.txt")
    gpu_reqs = _parse_requirements(REQUIREMENTS_DIR / "requirements-gpu.txt")

    assert "torch" in cpu_reqs
    assert "==" not in cpu_reqs["torch"]
    assert cpu_reqs["torch"].startswith("torch>=")
    assert "torch" not in gpu_reqs  # not re-pinned/overridden in the gpu file


def test_pillow_is_cpu_only_no_cuda_dependency_needed_for_synthetic_image_fixtures():
    cpu_reqs = _parse_requirements(REQUIREMENTS_DIR / "requirements-cpu.txt")
    assert "pillow" in cpu_reqs


def test_requirements_gpu_includes_requirements_cpu():
    gpu_text = (REQUIREMENTS_DIR / "requirements-gpu.txt").read_text()
    assert "-r requirements-cpu.txt" in gpu_text
