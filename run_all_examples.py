import subprocess
import glob
import os
import sys
import time

# Create log and result directories
os.makedirs("result/example_logs", exist_ok=True)

scripts = sorted(glob.glob("examples_py/*.py"))
print(f"Found {len(scripts)} scripts to execute.")

summary = []

# Environmental variables for sub-processes
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "1"
env["HF_HOME"] = os.path.abspath(".hf_cache")
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

for script_path in scripts:
    script_name = os.path.basename(script_path)
    log_path = f"result/example_logs/{script_name.replace('.py', '.log')}"
    
    print(f"\n==========================================")
    print(f"Running: {script_name}")
    print(f"==========================================")
    
    start_time = time.time()
    
    try:
        # Run script with a timeout of 180 seconds to prevent hangs on interactive notebooks
        result = subprocess.run(
            ["uv", "run", "python", script_path],
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
            cwd=os.getcwd()
        )
        duration = time.time() - start_time
        
        # Write stdout and stderr to log file
        with open(log_path, "w", encoding="utf-8") as lf:
            lf.write("=== STDOUT ===\n")
            lf.write(result.stdout)
            lf.write("\n=== STDERR ===\n")
            lf.write(result.stderr)
            
        if result.returncode == 0:
            status = "PASSED"
            error_msg = ""
            print(f"Status: {status} in {duration:.2f}s")
        else:
            status = "FAILED"
            # Get last few lines of stderr for the summary
            stderr_lines = result.stderr.strip().splitlines()
            error_msg = stderr_lines[-1] if stderr_lines else f"Exit code {result.returncode}"
            print(f"Status: {status} in {duration:.2f}s (Error: {error_msg})")
            
    except subprocess.TimeoutExpired as te:
        duration = time.time() - start_time
        status = "TIMEOUT"
        error_msg = "Timed out after 180s"
        with open(log_path, "w", encoding="utf-8") as lf:
            lf.write(f"=== TIMEOUT ===\nTimed out after 180 seconds.\n\n")
            lf.write("=== PARTIAL STDOUT ===\n")
            lf.write(te.stdout or "")
            lf.write("\n=== PARTIAL STDERR ===\n")
            lf.write(te.stderr or "")
        print(f"Status: {status} in {duration:.2f}s")
        
    except Exception as e:
        duration = time.time() - start_time
        status = "ERROR"
        error_msg = str(e)
        with open(log_path, "w", encoding="utf-8") as lf:
            lf.write(f"=== LAUNCH ERROR ===\n{error_msg}\n")
        print(f"Status: {status} in {duration:.2f}s")
        
    summary.append({
        "script": script_name,
        "status": status,
        "duration": f"{duration:.2f}s",
        "error": error_msg,
        "log": log_path
    })

# Write summary markdown report
summary_path = "result/examples_summary.md"
with open(summary_path, "w", encoding="utf-8") as sf:
    sf.write("# SAM3 Examples Execution Summary Report\n\n")
    sf.write("| Script Name | Status | Execution Time | Notes / Errors |\n")
    sf.write("| --- | --- | --- | --- |\n")
    for item in summary:
        sf.write(f"| [{item['script']}](file://{os.path.abspath('examples_py/' + item['script'])}) | **{item['status']}** | {item['duration']} | {item['error']} |\n")

print("\n==========================================")
print(f"All runs completed! Summary report written to {summary_path}")
print("==========================================")
