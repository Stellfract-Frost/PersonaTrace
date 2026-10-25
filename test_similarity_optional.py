"""The legacy similarity path must not require the optional vector extra."""

from personatrace import similarity
from personatrace.config import PersonaConfig


def test_similarity_falls_back_when_numpy_is_missing(monkeypatch):
    monkeypatch.setattr(similarity, "np", None)
    calc = similarity.SimilarityCalculator(PersonaConfig())
    calc.set_embed_fn(lambda texts: [[1.0, 0.0] for _ in texts])
    score = calc.compute("喜欢猫", "喜欢狗")
    assert 0.0 <= score <= 1.0
    assert score < 1.0


def test_similarity_still_works_with_numpy():
    calc = similarity.SimilarityCalculator(PersonaConfig())
    assert calc.compute("喜欢猫", "喜欢猫") == 1.0
