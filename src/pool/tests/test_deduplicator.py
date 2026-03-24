import pandas as pd
import pytest
from src.pool.deduplicator import Deduplicator


@pytest.fixture
def deduplicator():
    return Deduplicator()


class TestDeduplicator:

    def test_no_duplicates_unchanged(self, deduplicator):
        df = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [4.0, 5.0, 6.0], 'query_id': ['x', 'x', 'x']})
        result = deduplicator(df)
        assert len(result) == 3

    def test_exact_duplicates_removed(self, deduplicator):
        df = pd.DataFrame({'a': [1.0, 1.0, 2.0], 'b': [4.0, 4.0, 5.0], 'query_id': ['x', 'x', 'x']})
        result = deduplicator(df)
        assert len(result) == 2

    def test_query_id_not_used_for_dedup(self, deduplicator):
        # Same feature values but different query_ids — still a duplicate
        df = pd.DataFrame({'a': [1.0, 1.0], 'b': [2.0, 2.0], 'query_id': ['id1', 'id2']})
        result = deduplicator(df)
        assert len(result) == 1

    def test_different_query_id_same_features_keeps_first(self, deduplicator):
        df = pd.DataFrame({'a': [1.0, 1.0], 'b': [2.0, 2.0], 'query_id': ['id1', 'id2']})
        result = deduplicator(df)
        assert result.iloc[0]['query_id'] == 'id1'

    def test_custom_ignore_cols(self, deduplicator):
        df = pd.DataFrame({'a': [1.0, 1.0], 'b': [2.0, 2.0], 'meta': ['x', 'y']})
        result = deduplicator(df, ignore_cols=('meta',))
        assert len(result) == 1

    def test_index_is_reset(self, deduplicator):
        df = pd.DataFrame({'a': [1.0, 1.0, 2.0], 'query_id': ['x', 'x', 'x']})
        result = deduplicator(df)
        assert list(result.index) == list(range(len(result)))

    def test_empty_dataframe(self, deduplicator):
        df = pd.DataFrame({'a': [], 'b': [], 'query_id': []})
        result = deduplicator(df)
        assert len(result) == 0

    def test_no_ignore_cols(self, deduplicator):
        df = pd.DataFrame({'a': [1.0, 1.0], 'b': [2.0, 2.0]})
        result = deduplicator(df, ignore_cols=())
        assert len(result) == 1

    def test_all_duplicates(self, deduplicator):
        df = pd.DataFrame({'a': [5.0] * 10, 'query_id': ['x'] * 10})
        result = deduplicator(df)
        assert len(result) == 1
