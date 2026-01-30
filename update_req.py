import subprocess
import re

# Read requirements.txt
with open('requirements.txt', 'r') as f:
    lines = f.readlines()

packages = []
for line in lines:
    if '==' in line:
        package = line.split('==')[0].strip()
        packages.append(package)

# Upgrade each package
for package in packages:
    subprocess.run(['pip', 'install', '--upgrade', package])

# Get frozen versions
result = subprocess.run(['pip', 'freeze'], capture_output=True, text=True)
frozen = result.stdout.split('\n')

# Filter to only the packages
updated_lines = []
for line in frozen:
    if '==' in line:
        pkg = line.split('==')[0]
        if pkg in packages:
            updated_lines.append(line)

# Write back to requirements.txt
with open('requirements.txt', 'w') as f:
    f.write('\n'.join(updated_lines))
