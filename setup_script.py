import subprocess
import shutil
import os


def find_python_command():
    """Checks if 'python' or 'python3' is available and returns the correct command."""
    for cmd in ["python", "python3"]:
        try:
            subprocess.run(
                [cmd, "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            return cmd  # Return the working Python command
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    raise RuntimeError("Python is not installed or not found in PATH.")


def run_command(command, cwd=None):
    """Runs a shell command, prints output in real-time, and raises an error if it fails."""
    print(f"Running: {command}")
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
    )

    # Print output in real-time
    for line in process.stdout:
        print(line, end="")
    for line in process.stderr:
        print(line, end="")

    # Wait for process to finish
    process.wait()

    # Raise an error if the command failed
    if process.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {process.returncode}: {command}"
        )


# Step 1: Install dependencies for the client
print("Installing client dependencies...")
run_command("npm install", cwd="client")

# Step 2: Build the client
print("Building client...")
run_command("npm run build", cwd="client")

# Step 3: Install python dependencies
try:
    run_command("pip install .")
except:
    run_command("pip3 install .")

# Step 4: Copy client/public/ to src/client_public/
print("Copying client/public/ to src/client_public/...")
src_dir = os.path.join("client", "public")
dst_dir = os.path.join("src", "semantra", "client_public")

if os.path.exists(dst_dir):
    shutil.rmtree(dst_dir)
shutil.copytree(src_dir, dst_dir)

python_cmd = find_python_command()

# Step 5: Prompt user to run Semantra
print("Now run:")
print(f'\n\n\t{python_cmd} {os.path.join("src", "semantra", "semantra.py")}')
