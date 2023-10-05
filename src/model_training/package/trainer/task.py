import subprocess

# Define the shell script command as a string
shell_script_command = "./task-shell.sh"

# Use subprocess to run the shell script
process = subprocess.Popen(shell_script_command, shell=True)

# Wait for the shell script to complete
process.wait()

# Check the return code
if process.returncode == 0:
    print("Shell script executed successfully")
else:
    print(f"Shell script failed with return code {process.returncode}")