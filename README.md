# Polymarket Weather Alpha Tester

Data recorder and research tools for daily maximum-temperature markets in Amsterdam, London, Los Angeles and New York.

Discontinued. No tradable edge demonstrated. No real orders placed. Known model and execution issues are listed in [results](docs/RESULTS.md).

Python 3.11+.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[all,dev]"
playwright install chromium
Copy-Item .env.example .env
weather-alpha --help
python -m pytest -q
```

Historical data and fitted models are not included. Configure provider credentials in `.env` before collecting data.

MIT license.
