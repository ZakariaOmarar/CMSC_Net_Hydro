from __future__ import annotations


def test_flow_package_exports() -> None:
    from src.modeling.flow import train_and_calibrate_flow
    from src.modeling.flow.train import train_and_calibrate_flow as impl

    assert train_and_calibrate_flow is impl


def test_baselines_package_exports() -> None:
    from src.modeling.baselines import train_baseline_model
    from src.modeling.baselines.train import train_baseline_model as impl

    assert train_baseline_model is impl


def test_reporting_wrapper_exports() -> None:
    from src.modeling.reporting import collect_model_reports
    from src.modeling.reporting.report import collect_model_reports as impl

    assert collect_model_reports is impl


def test_mode_wrapper_exports() -> None:
    from src.modeling.mode import train_mode_classifier
    from src.modeling.mode.train import train_mode_classifier as impl

    assert train_mode_classifier is impl


def test_orchestration_wrapper_exports() -> None:
    from src.modeling.orchestration import main
    from src.modeling.orchestration.train_all import main as impl

    assert main is impl
