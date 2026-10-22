@echo off
rem Starts the Llmzempic web UI. Extra arguments are passed through (e.g. --port 8000 --no-browser).
cd /d "%~dp0"
".venv\Scripts\python.exe" -m llmzempic %*
