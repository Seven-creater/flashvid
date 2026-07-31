from pathlib import Path


def test_no_inner_llm_pruning_code():
    root = Path(__file__).parents[1] / "src"
    forbidden = ("fastv_prune", "pruning_layer", "llm_retention_ratio")
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in root.rglob("*.py")
    )
    for term in forbidden:
        assert term not in source

