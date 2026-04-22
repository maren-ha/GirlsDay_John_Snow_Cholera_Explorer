import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "serve_with_qr.py"


def load_qr_script():
    spec = importlib.util.spec_from_file_location("serve_with_qr", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_app_url_uses_http_ip_and_port():
    qr_script = load_qr_script()

    assert qr_script.build_app_url("192.168.1.23", 8501) == "http://192.168.1.23:8501"


def test_resolve_qr_url_prefers_explicit_external_url():
    qr_script = load_qr_script()

    assert (
        qr_script.resolve_qr_url(
            external_url="https://girlsday-cholera.streamlit.app",
            host_ip="192.168.1.23",
            port=8501,
        )
        == "https://girlsday-cholera.streamlit.app"
    )


def test_build_streamlit_command_exposes_app_on_local_network():
    qr_script = load_qr_script()

    command = qr_script.build_streamlit_command(port=8501, address="0.0.0.0")

    assert command[:4] == [qr_script.sys.executable, "-m", "streamlit", "run"]
    assert "app/streamlit_app.py" in command
    assert "--server.address" in command
    assert "0.0.0.0" in command
    assert "--server.port" in command
    assert "8501" in command
