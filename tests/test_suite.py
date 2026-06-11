import pytest

from agentdiff.suite import HoldoutBudgetExceeded, load_suite, record_holdout_reveal

SUITE_YAML = """
suite: test-suite
seed: 7
holdout_fraction: 0.4
max_holdout_reveals: 2
tasks:
  - id: t1
    input: { x: 1 }
  - id: t2
    input: { x: 2 }
  - id: t3
    input: { x: 3 }
  - id: t4
    input: { x: 4 }
  - id: t5
    input: { x: 5 }
"""


@pytest.fixture
def suite_path(tmp_path):
    p = tmp_path / "suite.yaml"
    p.write_text(SUITE_YAML, encoding="utf-8")
    return p


def test_hash_is_stable_and_content_sensitive(suite_path, tmp_path):
    s1 = load_suite(suite_path)
    s2 = load_suite(suite_path)
    assert s1.content_hash == s2.content_hash

    edited = tmp_path / "edited.yaml"
    edited.write_text(SUITE_YAML.replace("x: 5", "x: 6"), encoding="utf-8")
    assert load_suite(edited).content_hash != s1.content_hash


def test_split_is_deterministic_and_partitions(suite_path):
    s = load_suite(suite_path)
    dev = {t.id for t in s.split("dev")}
    holdout = {t.id for t in s.split("holdout")}
    assert dev | holdout == {t.id for t in s.tasks}
    assert dev & holdout == set()
    assert {t.id for t in load_suite(suite_path).split("dev")} == dev


def test_split_changes_with_seed(suite_path, tmp_path):
    edited = tmp_path / "seeded.yaml"
    edited.write_text(SUITE_YAML.replace("seed: 7", "seed: 8"), encoding="utf-8")
    a = {t.id for t in load_suite(suite_path).split("holdout")}
    b = {t.id for t in load_suite(edited).split("holdout")}
    # not guaranteed different for every seed pair, but these two differ
    assert a != b


def test_duplicate_ids_rejected(tmp_path):
    p = tmp_path / "dupe.yaml"
    p.write_text(SUITE_YAML.replace("id: t2", "id: t1"), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_suite(p)


def test_holdout_budget_enforced(suite_path):
    s = load_suite(suite_path)
    assert record_holdout_reveal(s) == 1
    assert record_holdout_reveal(s) == 2
    with pytest.raises(HoldoutBudgetExceeded):
        record_holdout_reveal(s)
    # override is allowed but still recorded
    assert record_holdout_reveal(s, override=True) == 3


def test_new_suite_version_gets_fresh_budget(suite_path):
    s = load_suite(suite_path)
    record_holdout_reveal(s)
    record_holdout_reveal(s)

    edited = suite_path.parent / "v2.yaml"
    edited.write_text(SUITE_YAML.replace("x: 5", "x: 50"), encoding="utf-8")
    s2 = load_suite(edited)
    assert record_holdout_reveal(s2) == 1
