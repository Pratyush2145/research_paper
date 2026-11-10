from importlib import import_module
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_expected_project_structure_exists():
    expected_paths = [
        "data/raw",
        "data/processed",
        "data/external",
        "notebooks",
        "src/data",
        "src/features",
        "src/models",
        "src/evaluation",
        "src/explainability",
        "artifacts",
        "app",
        "research",
    ]

    for relative_path in expected_paths:
        assert (PROJECT_ROOT / relative_path).is_dir(), relative_path


def test_essential_dependencies_and_project_packages_import():
    modules = [
        "numpy",
        "pandas",
        "sklearn",
        "matplotlib",
        "joblib",
        "src",
        "src.data.load_data",
        "src.data.preprocess",
        "src.features.url_features",
        "src.models.random_forest",
        "src.models.xgboost_model",
        "src.models.svm_model",
        "src.evaluation.metrics",
        "src.explainability.shap_analysis",
    ]

    for module_name in modules:
        import_module(module_name)
