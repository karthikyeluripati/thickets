import pytest

from thicket_runtime.committee_validation import checked_candidate


def test_mismatched_original_state_never_generates_and_still_restores():
    class Fake:
        def __init__(self):
            self.restored = False
        def rebase(self):
            pass
        def drift(self):
            return {'exact_base': True}
        def memory(self, reset=False):
            return {}
        def apply(self, *args):
            pass
        def fingerprint(self):
            return 'wrong-state'
        def infer(self):
            raise AssertionError('generation must not happen')
        def restore(self, *args):
            self.restored = True
    api = Fake()
    with pytest.raises(RuntimeError, match='BEFORE generation'):
        checked_candidate(api, {'candidate': {}, 'expected_state_id': 'original-state'}, [], None)
    assert api.restored
