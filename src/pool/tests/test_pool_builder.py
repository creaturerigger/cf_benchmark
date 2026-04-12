import uuid
from unittest.mock import MagicMock
import pandas as pd
import pytest
from src.pool.pool_builder import CFPoolBuilder


def _make_cf_result(rows: pd.DataFrame):
    """Build a mock DiCE-X result whose .to_dataframe() returns rows."""
    result = MagicMock()
    result.to_dataframe.return_value = rows
    return result


@pytest.fixture
def query():
    return pd.DataFrame({'age': [35.0], 'hours': [40.0]})


@pytest.fixture
def cf_rows():
    return pd.DataFrame({'age': [28.0, 45.0], 'hours': [35.0, 50.0]})


@pytest.fixture
def cf_method(cf_rows):
    mock = MagicMock()
    mock.generate.return_value = _make_cf_result(cf_rows)
    return mock


@pytest.fixture
def builder(cf_method, tmp_path, monkeypatch):
    monkeypatch.setattr(
        'src.pool.pool_builder.DefaultPaths.poolPath', tmp_path
    )
    return CFPoolBuilder(cf_method, runs=3, per_run=2, ds_name='test', save_interval=2)


class TestCFPoolBuilderInit:

    def test_cfs_filepath_contains_ds_name(self, builder):
        assert 'test' in builder.cfs_filepath.name

    def test_original_suffix_in_filename(self, builder):
        assert 'original' in builder.cfs_filepath.name

    def test_perturbed_suffix_in_filename(self, cf_method, tmp_path, monkeypatch):
        monkeypatch.setattr('src.pool.pool_builder.DefaultPaths.poolPath', tmp_path)
        b = CFPoolBuilder(cf_method, runs=1, per_run=1, ds_name='test',
                          perturbed=True)
        assert 'perturbed' in b.cfs_filepath.name

    def test_pool_dir_is_created(self, builder):
        assert builder.ds_pool_path.exists()


class TestBuildReturnValue:

    def test_returns_tuple(self, builder, query):
        result = builder.build(query)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_query_id_is_string(self, builder, query):
        qid, _ = builder.build(query)
        assert isinstance(qid, str)

    def test_returns_valid_uuid(self, builder, query):
        qid, _ = builder.build(query)
        uuid.UUID(qid)  # raises if not a valid UUID

    def test_accepts_existing_query_id(self, builder, query):
        existing_id = str(uuid.uuid4())
        returned_id, _ = builder.build(query, query_id=existing_id)
        assert returned_id == existing_id

    def test_returns_generated_cfs(self, builder, query):
        _, cfs_df = builder.build(query)
        assert isinstance(cfs_df, pd.DataFrame)
        assert len(cfs_df) > 0


class TestBuildFileOutput:

    def test_cfs_csv_is_created(self, builder, query):
        builder.build(query)
        assert builder.cfs_filepath.exists()

    def test_queries_csv_is_created(self, builder, query):
        builder.build(query)
        assert builder.queries_filepath.exists()

    def test_query_id_column_in_cfs(self, builder, query):
        builder.build(query)
        df = pd.read_csv(builder.cfs_filepath)
        assert 'query_id' in df.columns

    def test_query_id_column_in_queries(self, builder, query):
        builder.build(query)
        df = pd.read_csv(builder.queries_filepath)
        assert 'query_id' in df.columns

    def test_query_id_consistent_across_cfs(self, builder, query):
        qid, _ = builder.build(query)
        df = pd.read_csv(builder.cfs_filepath)
        assert (df['query_id'] == qid).all()

    def test_query_row_saved(self, builder, query):
        builder.build(query)
        df = pd.read_csv(builder.queries_filepath)
        assert len(df) == 1
        assert df.iloc[0]['age'] == pytest.approx(35.0)

    def test_cf_generate_called_runs_times(self, builder, query, cf_method):
        builder.build(query)
        assert cf_method.generate.call_count == 3


class TestBuildDeduplication:

    def test_duplicate_cfs_are_removed(self, cf_method, tmp_path, monkeypatch):
        monkeypatch.setattr('src.pool.pool_builder.DefaultPaths.poolPath', tmp_path)
        # All runs return the same two rows — should deduplicate to 2
        duplicate_rows = pd.DataFrame({'age': [28.0, 28.0], 'hours': [35.0, 35.0]})
        cf_method.generate.return_value = _make_cf_result(duplicate_rows)
        b = CFPoolBuilder(cf_method, runs=4, per_run=2, ds_name='test', save_interval=2)
        b.build(pd.DataFrame({'age': [35.0], 'hours': [40.0]}))
        df = pd.read_csv(b.cfs_filepath)
        assert len(df) == 1

    def test_two_builds_same_query_id_deduplicated(self, cf_method, tmp_path, monkeypatch):
        monkeypatch.setattr('src.pool.pool_builder.DefaultPaths.poolPath', tmp_path)
        b = CFPoolBuilder(cf_method, runs=2, per_run=2, ds_name='test', save_interval=10)
        qid = str(uuid.uuid4())
        b.build(pd.DataFrame({'age': [35.0], 'hours': [40.0]}), query_id=qid)
        b.build(pd.DataFrame({'age': [35.0], 'hours': [40.0]}), query_id=qid)
        df = pd.read_csv(b.cfs_filepath)
        # Both builds produce the same CF rows — dedup must keep only unique ones
        assert len(df) == len(df.drop_duplicates(subset=['age', 'hours']))
