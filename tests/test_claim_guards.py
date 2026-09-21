"""
Mutation tests for the prose-claim guards in test_claims.py.

A regression test that has never been seen to fail is indistinguishable from one
that cannot fail. These scenarios reintroduce, one at a time, every way a
reported number has drifted out of its documents, and require that the guard
responsible fails and that the others stay clean. A guard that silently stops
matching its claim is caught here rather than the next time a number moves.

Each scenario copies the prose documents to a throwaway directory, points
test_claims.ROOT at it and mutates the copy. The CSVs are never copied: the
guards keep deriving their expected values from the real data, which is what
makes them meaningful. Nothing under the working tree is opened for writing, and
every scenario checks that for itself.

Run with `python tests/test_claim_guards.py` or `pytest tests/`.
"""
import os
import re
import shutil
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_claims as tc

# The documents the guards read. Captured before any monkeypatching, so the
# isolation check below always compares against the real working tree.
REAL_ROOT = tc.ROOT
DOCS = ("PAPER.md", "README.md", "SUMMARY.md", "METHODS.md", "PAPER.html",
        "tutorial.ipynb")

GUARDS = {
    "mass": tc.test_rotor_mass_spread_claim_is_current_everywhere,
    "html_sync": tc.test_paper_html_is_synchronized_with_paper_md,
    "amp_gain": tc.test_rotor_amplification_and_loop_gain_claims_are_current,
    "withdrawn": tc.test_withdrawn_claims_do_not_return,
}


def _values(doc, pattern):
    """What a document currently quotes for one claim, read through the
    production pattern. Deriving every mutation target this way keeps the
    expected values in exactly one place, the CSV, and means a scenario cannot
    quietly go on testing a number the paper no longer reports."""
    return [m.group("value").strip() for m in re.finditer(pattern, tc._prose(doc), re.I)]


def _rewrite(tmp, doc, phrase, replacement, pattern=None):
    """Replace one occurrence of `phrase` in the throwaway copy of `doc`.

    Whitespace in the phrase matches any whitespace, because the documents wrap
    mid-sentence and a claim routinely straddles a line break. The phrase must
    occur exactly once, so rewording the source makes this harness fail loudly
    instead of mutating some other sentence. When `pattern` is given, the
    mutation is confirmed to have moved that claim's quoted values, which ties
    the scenario back to the production regex rather than to a literal.
    """
    path = os.path.join(tmp, doc)
    text = open(path, encoding="utf-8").read()
    rx = re.compile(r"\s+".join(re.escape(w) for w in phrase.split()))
    hits = rx.findall(text)
    assert len(hits) == 1, (
        f"mutation setup: {phrase!r} occurs {len(hits)} times in {doc}, expected 1. "
        "The document was reworded, so this scenario needs updating."
    )
    before = _values(doc, pattern) if pattern is not None else None
    # A lambda, not a string: a replacement is literal text, never a template.
    open(path, "w", encoding="utf-8").write(rx.sub(lambda _: replacement, text, count=1))
    if pattern is not None:
        assert _values(doc, pattern) != before, (
            f"mutation setup: rewriting {phrase!r} in {doc} did not change any "
            "value the production pattern reads, so this scenario proves nothing"
        )


# ── the mutations ────────────────────────────────────────────────────────────
# Anchors are derived from the documents through the production patterns.
# Replacements are the superseded values themselves, so each scenario exercises
# both nets in assert_claim: the value check and the stale-pattern check.

def _stale_mass_in_paper(tmp):
    """The regression that started this. The body of PAPER.md was corrected and
    one sentence near the end kept the pre-convergence wording, which a
    presence-only assertion could not see."""
    v = _values("PAPER.md", tc.CLAIM_MASS_SPREAD)[2]
    _rewrite(tmp, "PAPER.md",
             f"worth about {v} kilograms of take-off mass.",
             "worth about three kilograms of take-off mass.",
             tc.CLAIM_MASS_SPREAD)


def _stale_mass_in_readme(tmp):
    """The same number in its numeric form, in a different document."""
    v = _values("README.md", tc.CLAIM_MASS_SPREAD)[0]
    _rewrite(tmp, "README.md", f"{v} kg on a", "3.0 kg on a", tc.CLAIM_MASS_SPREAD)


def _delete_mass_in_summary(tmp):
    """A claim that disappears is as much a drift as one that goes stale."""
    v = _values("SUMMARY.md", tc.CLAIM_MASS_SPREAD)[0]
    _rewrite(tmp, "SUMMARY.md",
             f"worth about {v} kilograms of take-off mass",
             "no longer quoted here", tc.CLAIM_MASS_SPREAD)


def _duplicate_mass_in_paper(tmp):
    """And so is one that gets stated twice: the second copy is where a stale
    value hides next time."""
    v = _values("PAPER.md", tc.CLAIM_MASS_SPREAD)[2]
    sentence = f"worth about {v} kilograms of take-off mass."
    _rewrite(tmp, "PAPER.md", sentence, f"{sentence} It is again {sentence}",
             tc.CLAIM_MASS_SPREAD)


def _desync_mass(tmp):
    """PAPER.md edited, build_paper_html.py not re-run."""
    v = _values("PAPER.md", tc.CLAIM_MASS_SPREAD)[0]
    _rewrite(tmp, "PAPER.md", f"worth {v} kilograms on a", "worth four kilograms on a",
             tc.CLAIM_MASS_SPREAD)


def _stale_amp_range(tmp):
    v = _values("README.md", tc.CLAIM_AMP_RANGE)[0]
    _rewrite(tmp, "README.md", f"amplification factor is {v}",
             "amplification factor is 0.62-0.68", tc.CLAIM_AMP_RANGE)


def _stale_amp_percentile(tmp):
    """The second statement of the amplification factor, one value per
    percentile. This is the shape the range pattern cannot see, so it carries
    its own guard; 0.72 is the value that guard's stale pattern names."""
    v = _values("PAPER.md", tc.CLAIM_AMP_PERCENTILE)[0]
    _rewrite(tmp, "PAPER.md", f"section drag is {v}",
             "section drag is " + re.sub(r"\d\.\d+", "0.72", v, count=1),
             tc.CLAIM_AMP_PERCENTILE)


def _stale_gain_paren(tmp):
    v = _values("README.md", tc.CLAIM_GAIN_PAREN)[0]
    _rewrite(tmp, "README.md", f"doubles ({v}x)", "doubles (1.9-2.5x)",
             tc.CLAIM_GAIN_PAREN)


def _stale_gain_prose(tmp):
    """The same loop gain in the paper's own wording, which the parenthesised
    README form does not match."""
    v = _values("PAPER.md", tc.CLAIM_GAIN_PROSE)[0]
    _rewrite(tmp, "PAPER.md", f"multiplies the power error by {v} times",
             "multiplies the power error by 1.9 to 2.5 times", tc.CLAIM_GAIN_PROSE)


def _stale_html_percentile(tmp):
    """The generated file goes stale on its own: PAPER.md is current, PAPER.html
    is not, which is what happens when the builder is not re-run."""
    v = _values("PAPER.html", tc.CLAIM_AMP_PERCENTILE)[0]
    _rewrite(tmp, "PAPER.html", f"section drag is {v}",
             "section drag is " + re.sub(r"\d\.\d+", "0.72", v, count=1),
             tc.CLAIM_AMP_PERCENTILE)


def _delete_amp_in_html(tmp):
    """A claim dropped from the generated file only."""
    v = _values("PAPER.html", tc.CLAIM_AMP_RANGE)[0]
    _rewrite(tmp, "PAPER.html", f"take-off weight is {v}",
             "take-off weight is steady", tc.CLAIM_AMP_RANGE)


def _delete_gain_in_paper(tmp):
    """And one dropped from the source, which desynchronises the other way."""
    v = _values("PAPER.md", tc.CLAIM_GAIN_PROSE)[0]
    _rewrite(tmp, "PAPER.md", f"The loop multiplies the power error by {v} times.",
             "The loop matters.", tc.CLAIM_GAIN_PROSE)


def _revive_variance_claim(tmp):
    """The control. A withdrawn claim returning must fire the withdrawn-claims
    guard and nothing else, which is how these scenarios show they are not
    simply failing everything."""
    with open(os.path.join(tmp, "METHODS.md"), "a", encoding="utf-8") as f:
        f.write("\n\nThe signed errors track each other, explaining 86% of the "
                "variance.\n")


SCENARIOS = [
    ("baseline, unmutated", None, set()),
    ("stale 'three kilograms' returns to PAPER.md", _stale_mass_in_paper,
     {"mass", "html_sync"}),
    ("README quotes the superseded 3.0 kg", _stale_mass_in_readme, {"mass"}),
    ("mass claim deleted from SUMMARY.md", _delete_mass_in_summary, {"mass"}),
    ("mass claim duplicated in PAPER.md", _duplicate_mass_in_paper,
     {"mass", "html_sync"}),
    ("PAPER.md mass edited, HTML not rebuilt", _desync_mass, {"mass", "html_sync"}),
    ("stale amplification range in README", _stale_amp_range, {"amp_gain"}),
    ("stale per-percentile amplification in PAPER.md", _stale_amp_percentile,
     {"amp_gain", "html_sync"}),
    ("stale loop gain, parenthesised, in README", _stale_gain_paren, {"amp_gain"}),
    ("stale loop gain in the paper's own wording", _stale_gain_prose,
     {"amp_gain", "html_sync"}),
    ("generated HTML stale while PAPER.md is current", _stale_html_percentile,
     {"amp_gain", "html_sync"}),
    ("amplification range deleted from the generated HTML", _delete_amp_in_html,
     {"amp_gain", "html_sync"}),
    ("loop-gain sentence deleted from PAPER.md", _delete_gain_in_paper,
     {"amp_gain", "html_sync"}),
    ("withdrawn '86% of the variance' returns", _revive_variance_claim, {"withdrawn"}),
]


def _fingerprint():
    """Size and modification time of the real documents. A scenario that escaped
    its temp directory moves one of these."""
    return {d: os.stat(os.path.join(REAL_ROOT, d))[:10:9]
            for d in DOCS}


def _fired(mutate):
    """Apply one mutation to throwaway copies and report which guards failed."""
    untouched = _fingerprint()
    with tempfile.TemporaryDirectory(prefix="claim-mutation-") as tmp:
        for doc in DOCS:
            shutil.copy(os.path.join(REAL_ROOT, doc), os.path.join(tmp, doc))
        real_root, tc.ROOT = tc.ROOT, tmp
        try:
            assert not tc.DATA.startswith(tmp), \
                "the CSVs must still resolve to the real data directory"
            if mutate is not None:
                mutate(tmp)
            fired = {}
            for name, guard in GUARDS.items():
                try:
                    guard()
                except AssertionError as exc:
                    fired[name] = str(exc).splitlines()[0][:140]
            return fired
        finally:
            tc.ROOT = real_root
            assert _fingerprint() == untouched, \
                "a mutation wrote through to the real documents"


def _check(label, mutate, expected):
    fired = _fired(mutate)
    detail = "".join(f"\n    {n}: {m}" for n, m in fired.items())
    assert set(fired) == expected, (
        f"{label}: expected {sorted(expected) or ['no guard']} to fail, "
        f"got {sorted(fired) or ['none']}.{detail}"
    )
    return fired


@pytest.mark.parametrize("label,mutate,expected", SCENARIOS,
                         ids=[s[0] for s in SCENARIOS])
def test_guard_fires_for_exactly_its_own_mutation(label, mutate, expected):
    _check(label, mutate, expected)


def test_every_guard_is_exercised_by_some_scenario():
    """A guard nobody mutates is a guard nobody has checked. This fails when a
    new claim guard is added to test_claims.py without a scenario here."""
    covered = set().union(*(expected for _, _, expected in SCENARIOS))
    assert covered == set(GUARDS), \
        f"guards with no mutation scenario: {sorted(set(GUARDS) - covered)}"


if __name__ == "__main__":
    fails = 0
    for label, mutate, expected in SCENARIOS:
        try:
            _check(label, mutate, expected)
            print(f"PASS  {label}")
        except AssertionError as e:
            fails += 1
            print(f"FAIL  {e}")
    try:
        test_every_guard_is_exercised_by_some_scenario()
        print("PASS  every guard is exercised by some scenario")
    except AssertionError as e:
        fails += 1
        print(f"FAIL  {e}")
    print(f"\n{len(SCENARIOS) + 1 - fails}/{len(SCENARIOS) + 1} checks behaved as intended")
    sys.exit(1 if fails else 0)
