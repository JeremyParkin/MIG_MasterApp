from __future__ import annotations

import random
import sys

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from scipy import sparse

from bounded_token_support import original_token_extraction, reference
from processing import standard_cleaning, story_grouping
from test_agility_fixture_spine import FIXTURE_DIR, _run_basic_cleaning_like_page


# Include every whitespace character recognized by this Python runtime, including
# ASCII control separators, NBSP, narrow NBSP, and the Unicode space separators.
WHITESPACE = tuple(chr(value) for value in range(sys.maxunicode + 1) if chr(value).isspace())
TOKEN_COUNTS = [0, 1, 11, 12, 13, 18, 19, 20, 21, 100, 20_000]


@pytest.mark.parametrize("separator", WHITESPACE, ids=lambda s: f"U+{ord(s):04X}")
@pytest.mark.parametrize("count", TOKEN_COUNTS)
def test_prefix_matches_full_split_at_boundaries(separator, count):
    text = separator * 3 + separator.join(["café東京🙂"] * count) + separator * 2
    assert standard_cleaning._leading_token_tuple(text) == reference._leading_token_tuple(text)


@pytest.mark.parametrize("limit", [-20, -3, -1, 0, 1, 11, 12, 13, 20])
@pytest.mark.parametrize("text", [None, "", " \t\n", "one", "one\u200btwo\ufeffthree", "word " * 100])
def test_prefix_preserves_empty_inputs_non_whitespace_and_custom_limits(text, limit):
    assert standard_cleaning._leading_token_tuple(text, limit) == reference._leading_token_tuple(text, limit)


@pytest.mark.parametrize("separator", WHITESPACE, ids=lambda s: f"U+{ord(s):04X}")
@pytest.mark.parametrize("count", TOKEN_COUNTS)
def test_weak_snippet_edge_decision_and_vectorizer_input_match(monkeypatch, separator, count):
    snippet = separator * 2 + separator.join(["café東京🙂"] * count) + separator * 3
    frame = pd.DataFrame({
        "Normalized Headline": ["same headline", "same headline"],
        "Normalized Snippet": [snippet, "strong " * 20],
    }, index=pd.Index([8, 2], name="source"))
    before = frame.copy(deep=True)
    vectorizer_inputs = []

    class Vectorizer:
        def fit_transform(self, texts):
            vectorizer_inputs.append(texts.copy())
            return sparse.eye(2, format="csr")

    class Neighbors:
        def __init__(self, **kwargs):
            assert kwargs == dict(metric="cosine", radius=1 - 0.935, algorithm="brute", n_jobs=-1)

        def fit(self, matrix):
            pass

        def radius_neighbors_graph(self, matrix, **kwargs):
            # A 0.94 cosine edge survives only when both snippets have >=20
            # tokens. This exercises the real edge filtering and components.
            return sparse.csr_matrix([[0.0, 0.06], [0.06, 0.0]])

    for module in [reference, story_grouping]:
        monkeypatch.setattr(module, "TfidfVectorizer", Vectorizer)
        monkeypatch.setattr(module, "NearestNeighbors", Neighbors)
    expected = reference.cluster_similar_stories(frame, 0.935)
    actual = story_grouping.cluster_similar_stories(frame, 0.935)
    assert_frame_equal(actual, expected, check_exact=True)
    assert actual["Group ID"].nunique() == (2 if count < 20 else 1)
    pd.testing.assert_series_equal(vectorizer_inputs[0], vectorizer_inputs[1], check_exact=True)
    assert vectorizer_inputs[1].iloc[0] == "same headline " + snippet
    assert_frame_equal(frame, before, check_exact=True)


@pytest.mark.parametrize("seed", range(12))
def test_randomized_unicode_prefixes_and_thresholds_match(seed):
    rng = random.Random(seed)
    samples = []
    for _ in range(150):
        words = [rng.choice(["café", "東京", "🙂", "a\u200bb", "a\ufeffb", "punctuation!"])
                 for _ in range(rng.randrange(100))]
        text = rng.choice(WHITESPACE) + "".join(word + rng.choice(WHITESPACE) * rng.randrange(1, 4)
                                                for word in words)
        samples.append(text)
        assert standard_cleaning._leading_token_tuple(text) == reference._leading_token_tuple(text)
    snippets = pd.Series(samples)
    original = snippets.str.split().str.len().to_numpy(dtype=np.int32) < 20
    bounded = snippets.str.split(n=19).str.len().to_numpy(dtype=np.int32) < 20
    np.testing.assert_array_equal(bounded, original)


@pytest.mark.parametrize("fixture", ["agility_golden_corpus.csv", "agility_malformed_inputs.csv"])
def test_full_pipeline_outputs_match_original_exactly(fixture):
    raw = pd.read_csv(FIXTURE_DIR / fixture)
    _assert_pipeline_equivalence(raw)


@pytest.mark.parametrize("seed", range(8))
def test_full_pipeline_with_randomized_long_unicode_and_missing_snippets(seed):
    rng = random.Random(seed)
    raw = pd.read_csv(FIXTURE_DIR / "agility_golden_corpus.csv")
    snippets = [None, "", " \t\n", "\u200b\ufeff", "word " * 8_000]
    snippets += [separator.join(["coverage"] * count)
                 for separator in WHITESPACE for count in [11, 12, 13, 19, 20, 21]]
    raw["Coverage Snippet"] = [rng.choice(snippets) for _ in range(len(raw))]
    # Exercise undated comparison paths as well as the fixture's broadcast,
    # social, syndication, and tied representative-selection cases.
    raw.loc[raw.index % 5 == seed % 5, "Published Date"] = None
    _assert_pipeline_equivalence(raw)


def _assert_pipeline_equivalence(raw):
    before = raw.copy(deep=True)
    with original_token_extraction():
        expected = _run_basic_cleaning_like_page(raw)
    actual = _run_basic_cleaning_like_page(raw)
    assert actual.keys() == expected.keys()
    for key in expected:
        assert_frame_equal(actual[key], expected[key], check_exact=True, obj=key)
    assert_frame_equal(raw, before, check_exact=True)
