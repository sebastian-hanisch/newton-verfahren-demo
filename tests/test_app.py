from streamlit.testing.v1 import AppTest

_APP_TIMEOUT = 60


def _fresh():
    at = AppTest.from_file("../app.py", default_timeout=_APP_TIMEOUT)
    at.run()
    return at


def test_app_runs_without_exception():
    at = _fresh()
    assert not at.exception


def test_footer_is_present():
    at = _fresh()
    captions = [c.value for c in at.caption]
    assert any("Sebastian Hanisch" in c and "Über mich" in c for c in captions)


def test_preset_quadratik_ein_schritt():
    at = _fresh()
    btn = [b for b in at.button if b.label == "Quadratik — ein Schritt"][0]
    btn.click().run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Iterationen"] == "1"


def test_preset_log_beule_divergiert():
    at = _fresh()
    btn = [b for b in at.button if b.label == "Log-Beule — divergiert"][0]
    btn.click().run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert "divergiert" in metrics["Konvergiert?"]


def test_switching_functions_does_not_crash():
    at = _fresh()
    radio = [r for r in at.radio if r.label == "Funktion"][0]
    radio.set_value("rosenbrock").run()
    assert not at.exception
    radio = [r for r in at.radio if r.label == "Funktion"][0]
    radio.set_value("log_beule").run()
    assert not at.exception
