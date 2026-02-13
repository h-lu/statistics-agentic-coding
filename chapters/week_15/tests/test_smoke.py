"""
Smoke tests for Week 15: 高级统计计算

Basic tests to verify the testing infrastructure is working.
These are minimal sanity checks, not comprehensive tests.
"""

import pytest
import numpy as np
import pandas as pd


class TestSmokeImports:
    """Test that required packages can be imported."""

    def test_import_numpy(self):
        """Verify numpy can be imported."""
        import numpy
        assert numpy is not None

    def test_import_pandas(self):
        """Verify pandas can be imported."""
        import pandas
        assert pandas is not None

    def test_import_sklearn_pca(self):
        """Verify sklearn PCA can be imported."""
        from sklearn.decomposition import PCA
        assert PCA is not None

    def test_import_sklearn_kmeans(self):
        """Verify sklearn KMeans can be imported."""
        from sklearn.cluster import KMeans
        assert KMeans is not None

    def test_import_sklearn_metrics(self):
        """Verify sklearn metrics can be imported."""
        from sklearn.metrics import silhouette_score
        assert silhouette_score is not None

    def test_import_scipy_stats(self):
        """Verify scipy.stats can be imported."""
        from scipy import stats
        assert stats is not None


class TestSmokeBasicPCA:
    """Smoke tests for basic PCA functionality."""

    def test_pca_can_be_instantiated(self):
        """Verify PCA can be created."""
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        assert pca is not None
        assert pca.n_components == 2

    def test_pca_can_fit_simple_data(self):
        """Verify PCA can fit simple data."""
        from sklearn.decomposition import PCA

        # Create simple data
        X = np.random.randn(50, 5)

        # Fit PCA
        pca = PCA(n_components=2)
        pca.fit(X)

        # Check it has fitted
        assert hasattr(pca, 'components_')
        assert hasattr(pca, 'explained_variance_ratio_')

    def test_pca_can_transform_data(self):
        """Verify PCA can transform data."""
        from sklearn.decomposition import PCA

        X = np.random.randn(50, 5)
        pca = PCA(n_components=2)
        X_transformed = pca.fit_transform(X)

        assert X_transformed.shape == (50, 2)


class TestSmokeBasicClustering:
    """Smoke tests for basic clustering functionality."""

    def test_kmeans_can_be_instantiated(self):
        """Verify KMeans can be created."""
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=3, random_state=42)
        assert kmeans is not None
        assert kmeans.n_clusters == 3

    def test_kmeans_can_fit_simple_data(self):
        """Verify KMeans can fit simple data."""
        from sklearn.cluster import KMeans

        X = np.random.randn(50, 2)

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(X)

        # Check it has fitted
        assert hasattr(kmeans, 'cluster_centers_')
        assert hasattr(kmeans, 'labels_')
        assert len(kmeans.labels_) == 50

    def test_kmeans_can_predict(self):
        """Verify KMeans can predict cluster labels."""
        from sklearn.cluster import KMeans

        X = np.random.randn(50, 2)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(X)

        # Predict on same data
        labels = kmeans.predict(X)

        assert len(labels) == 50
        assert all(0 <= label < 3 for label in labels)


class TestSmokeBasicMetrics:
    """Smoke tests for basic metrics."""

    def test_silhouette_score_can_be_computed(self):
        """Verify silhouette score can be computed."""
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        X = np.random.randn(50, 2)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        score = silhouette_score(X, labels)

        assert isinstance(score, (float, np.floating))
        assert -1 <= score <= 1


class TestSmokeStreamingStats:
    """Smoke tests for streaming statistics."""

    def test_online_mean_basic(self):
        """Verify online mean calculation works."""
        # Simple online mean implementation
        class OnlineMean:
            def __init__(self):
                self.n = 0
                self.sum = 0.0

            def update(self, x):
                self.n += 1
                self.sum += x
                return self.mean()

            def mean(self):
                return self.sum / self.n if self.n > 0 else 0.0

        # Test
        om = OnlineMean()
        data = [1.0, 2.0, 3.0, 4.0, 5.0]

        for x in data:
            om.update(x)

        expected = np.mean(data)
        assert abs(om.mean() - expected) < 1e-10

    def test_online_variance_basic(self):
        """Verify online variance calculation works."""
        # Welford's algorithm
        class OnlineVar:
            def __init__(self):
                self.n = 0
                self.mean = 0.0
                self.M2 = 0.0

            def update(self, x):
                self.n += 1
                delta = x - self.mean
                self.mean += delta / self.n
                delta2 = x - self.mean
                self.M2 += delta * delta2

            def variance(self):
                return self.M2 / self.n if self.n > 0 else 0.0

        # Test
        ov = OnlineVar()
        data = np.random.randn(100)

        for x in data:
            ov.update(x)

        expected = np.var(data, ddof=0)
        assert abs(ov.variance() - expected) < 1e-10


class TestSmokeABTesting:
    """Smoke tests for A/B testing."""

    def test_ttest_can_be_run(self):
        """Verify t-test can be run."""
        from scipy import stats

        group_A = np.random.normal(100, 20, 100)
        group_B = np.random.normal(105, 20, 100)

        t_stat, p_value = stats.ttest_ind(group_A, group_B)

        assert isinstance(p_value, (float, np.floating))
        assert 0 <= p_value <= 1

    def test_chisquare_can_be_run(self):
        """Verify chi-square test can be run."""
        from scipy import stats

        observed = np.array([50, 50])
        expected = np.array([50, 50])

        chi2, p_value = stats.chisquare(observed, expected)

        assert isinstance(p_value, (float, np.floating))
        assert 0 <= p_value <= 1


class TestSmokeFixtures:
    """Smoke tests for test fixtures."""

    def test_simple_2d_data_fixture(self, simple_2d_data):
        """Verify simple_2d_data fixture works."""
        assert isinstance(simple_2d_data, pd.DataFrame)
        assert simple_2d_data.shape == (200, 2)
        assert 'feature1' in simple_2d_data.columns
        assert 'feature2' in simple_2d_data.columns

    def test_high_dim_data_fixture(self, high_dim_data):
        """Verify high_dim_data fixture works."""
        assert isinstance(high_dim_data, pd.DataFrame)
        assert high_dim_data.shape[0] == 100  # n_samples
        assert high_dim_data.shape[1] == 50  # n_features

    def test_well_separated_clusters_fixture(self, well_separated_clusters):
        """Verify well_separated_clusters fixture works."""
        assert isinstance(well_separated_clusters, pd.DataFrame)
        assert 'x' in well_separated_clusters.columns
        assert 'y' in well_separated_clusters.columns
        assert 'true_label' in well_separated_clusters.columns
        assert len(set(well_separated_clusters['true_label'])) == 3

    def test_streaming_data_fixture(self, streaming_data):
        """Verify streaming_data fixture works."""
        assert isinstance(streaming_data, np.ndarray)
        assert len(streaming_data) == 1000

    def test_ab_test_data_fixtures(self, ab_test_data_significant, ab_test_data_no_effect):
        """Verify A/B test data fixtures work."""
        # Significant data
        assert isinstance(ab_test_data_significant, pd.DataFrame)
        assert 'group' in ab_test_data_significant.columns
        assert 'value' in ab_test_data_significant.columns
        assert set(ab_test_data_significant['group']) == {'A', 'B'}

        # No effect data
        assert isinstance(ab_test_data_no_effect, pd.DataFrame)
        assert 'group' in ab_test_data_no_effect.columns
        assert 'value' in ab_test_data_no_effect.columns


class TestSmokeDataStructures:
    """Smoke tests for data structure expectations."""

    def test_dataframe_creation(self):
        """Verify DataFrame can be created."""
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3, 2)

    def test_numpy_array_creation(self):
        """Verify numpy array can be created."""
        arr = np.array([[1, 2], [3, 4], [5, 6]])
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (3, 2)

    def test_basic_dataframe_operations(self):
        """Verify basic DataFrame operations work."""
        df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B'],
            'value': [1.0, 2.0, 3.0, 4.0]
        })

        # Filter
        group_a = df[df['group'] == 'A']
        assert len(group_a) == 2

        # Get values
        values_a = group_a['value'].values
        assert isinstance(values_a, np.ndarray)


class TestSmokeConceptual:
    """Smoke tests for conceptual understanding."""

    def test_pca_reduces_dimensions(self):
        """Conceptual: PCA reduces dimensionality."""
        from sklearn.decomposition import PCA

        X = np.random.randn(50, 10)
        pca = PCA(n_components=5)
        X_transformed = pca.fit_transform(X)

        assert X_transformed.shape[1] < X.shape[1]

    def test_clustering_groups_similar_data(self):
        """Conceptual: Clustering groups similar data points."""
        from sklearn.cluster import KMeans

        # Create 3 obvious clusters
        cluster1 = np.random.randn(20, 2) * 0.5 + [0, 0]
        cluster2 = np.random.randn(20, 2) * 0.5 + [10, 10]
        cluster3 = np.random.randn(20, 2) * 0.5 + [-10, 10]

        X = np.vstack([cluster1, cluster2, cluster3])

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        # Should find 3 clusters
        assert len(set(labels)) == 3

    def test_streaming_updates_incrementally(self):
        """Conceptual: Streaming updates incrementally."""
        class OnlineMean:
            def __init__(self):
                self.count = 0

            def update(self, x):
                self.count += 1
                return self.count

        om = OnlineMean()
        for i in range(10):
            om.update(i)

        # Should have processed 10 updates
        assert om.count == 10

    def test_ab_test_compares_groups(self):
        """Conceptual: A/B test compares two groups."""
        from scipy import stats

        # Same groups -> no significant difference
        group1 = np.random.normal(100, 20, 100)
        group2 = np.random.normal(100, 20, 100)

        t_stat, p_value = stats.ttest_ind(group1, group2)

        # p should be > 0.05 most of the time
        # (This is probabilistic, so we just check it runs)
        assert isinstance(p_value, (float, np.floating))
