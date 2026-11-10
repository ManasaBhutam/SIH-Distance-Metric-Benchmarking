
from pathlib import Path

folder = Path("data/raw")

print("Looking inside:", folder.resolve())

if not folder.exists():
    print("ERROR: Folder does not exist.")
else:
    files = list(folder.iterdir())
    if files:
        print("Files found:")
        for file in files:
            print("-", file.name)
    else:
        print("Folder is empty. Copy the extracted dataset files here.")
