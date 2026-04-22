#!/usr/bin/env python3
"""Run the Streamlit app on the local network and generate a QR code."""

import argparse
from pathlib import Path
import socket
import subprocess
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_QR_PATH = ROOT_DIR / "assets" / "app-qr.png"


def find_lan_ip():
    """Return the likely LAN IP address without sending network traffic."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return "127.0.0.1"


def build_app_url(host_ip, port):
    return f"http://{host_ip}:{port}"


def build_streamlit_command(port, address):
    return [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app/streamlit_app.py",
        "--server.address",
        address,
        "--server.port",
        str(port),
    ]


def write_qr_png(url, output_path):
    try:
        import qrcode
    except ImportError as exc:
        raise RuntimeError(
            "The QR helper needs the qrcode package. Install dependencies with "
            "`pip install -r requirements.txt` first."
        ) from exc

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = qrcode.make(url)
    image.save(output_path)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Start the Streamlit app for devices on the same Wi-Fi and create a QR code."
    )
    parser.add_argument("--port", type=int, default=8501, help="Port for the Streamlit app.")
    parser.add_argument(
        "--address",
        default="0.0.0.0",
        help="Address Streamlit binds to. Keep 0.0.0.0 for same-Wi-Fi access.",
    )
    parser.add_argument(
        "--host-ip",
        default=None,
        help="IP address to encode in the QR code. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--qr-path",
        default=str(DEFAULT_QR_PATH),
        help="Where to save the QR code PNG.",
    )
    parser.add_argument(
        "--no-launch",
        action="store_true",
        help="Only create the QR code and print the command; do not start Streamlit.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    host_ip = args.host_ip or find_lan_ip()
    url = build_app_url(host_ip, args.port)
    qr_path = write_qr_png(url, args.qr_path)
    command = build_streamlit_command(args.port, args.address)

    print("\nLocal classroom app link")
    print(f"URL: {url}")
    print(f"QR code: {qr_path}")
    print("\nStudents need to be on the same Wi-Fi/network as this computer.")
    print("If the phone cannot open the link, check macOS firewall and the Wi-Fi network.")

    if args.no_launch:
        print("\nStart command:")
        print(" ".join(command))
        return 0

    print("\nStarting Streamlit. Leave this terminal window open during the session.")
    return subprocess.call(command, cwd=ROOT_DIR)


if __name__ == "__main__":
    raise SystemExit(main())
