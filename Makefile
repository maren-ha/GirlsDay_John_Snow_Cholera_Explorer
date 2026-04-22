# Simple automation for devs and educators

.PHONY: help dev app qr notebook clean

help:
	@echo "Targets:"
	@echo "  make dev       - See README for conda/pip setup"
	@echo "  make app       - Run Streamlit app"
	@echo "  make qr        - Run Streamlit app and create a same-Wi-Fi QR code"
	@echo "  make notebook  - Launch Jupyter Lab"
	@echo "  make clean     - Remove Python caches"

dev:
	@echo "👉 Create the conda env with: conda env create -f environment.yml"
	@echo "👉 Then activate it:         conda activate cholera-demo"
	@echo "👉 Or use pip:               python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt"

app:
	streamlit run app/streamlit_app.py

qr:
	python scripts/serve_with_qr.py

notebook:
	jupyter lab

clean:
	rm -rf __pycache__ */__pycache__ .ipynb_checkpoints
