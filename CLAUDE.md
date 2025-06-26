# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a minimal FastAPI application with two endpoints:
- Root endpoint `/` that returns a "Hello World" message
- Parameterized endpoint `/hello/{name}` that greets a specific name

## Development Commands

### Running the Application
```bash
# Install FastAPI and uvicorn if not already installed
pip install fastapi uvicorn

# Run the development server
uvicorn main:app --reload

# Run on specific host/port
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Testing
- HTTP test file available at `test_main.http` for testing endpoints
- Default server runs on `http://127.0.0.1:8000`

## Architecture

- Single-file FastAPI application in `main.py`
- No database or external dependencies beyond FastAPI
- Simple REST API structure with async endpoint handlers