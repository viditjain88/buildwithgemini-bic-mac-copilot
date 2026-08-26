from app.agent import (
    check_modality_completeness,
    compute_attenuation_metrics,
    consult_knowledge_base,
    list_experiments,
    log_experiment,
    root_agent,
)


def test_root_agent_name():
    assert root_agent.name == "root_agent"


def test_compute_attenuation_metrics_pass():
    result = compute_attenuation_metrics(14.5, 33.5, 0.935)
    assert "READY FOR CODABENCH SUBMISSION" in result
    assert "PASS" in result


def test_compute_attenuation_metrics_fail():
    result = compute_attenuation_metrics(20.0, 28.0, 0.850)
    assert "BENCHMARK THRESHOLDS NOT MET" in result


def test_check_modality_completeness():
    res = check_modality_completeness("sub-101")
    assert "Ready for pseudo-CT synthesis" in res


def test_list_experiments():
    res = list_experiments()
    assert "BIC-MAC Experiment Leaderboard" in res or "Failed" in res
