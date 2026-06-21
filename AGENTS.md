# Agent Guidelines & Rules

Please follow these instructions strictly for all development tasks in this workspace:

## 1. Background Service Shell Scripts
Whenever implementing a background-running Python script or service, you must create the following management scripts in the workspace:
- **`install.sh`**: Installs dependencies and prepares the workspace using `uv`.
- **`start.sh`**: Launches the Python script/service in the background (e.g., using `nohup`, output redirection) and logs the output.
- **`stop.sh`**: Safely stops the running background process.
- **`monitor.sh`**: Checks if the background process is running, monitors resources/logs, and prints the current status.

## 2. Python Environment Management with `uv`
- Always use **`uv`** to manage Python environments and execute all Python scripts.
- Do NOT use bare `pip`, `python`, or `python3` commands directly.
- Use only the standard `uv` commands:
  - `uv init` (to initialize Python project configuration)
  - `uv add` (to add dependencies)
  - `uv sync` (to sync environment dependencies)
  - `uv run` (to execute Python scripts)

## 3. Git Submodules Constraints
- **NEVER** add, edit, or remove Git submodules.
- Keep the existing submodule configurations (including `.gitmodules` and folders like `sam3`) completely unmodified.

## 4. Planning File Location
- Before initiating any code implementation, always write the plan in the `/plan/` directory (located at the workspace root).

## 5. Blocking Issues Logging
- Write any blocking issues, critical errors, or design blockers into files in the `/issues/` directory (located at the workspace root).

## 6. Git Operations
- Do NOT commit or push changes to git, except when explicitly instructed by the user.
