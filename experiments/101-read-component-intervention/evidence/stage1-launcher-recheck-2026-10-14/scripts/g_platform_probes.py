"""Platform facts the hook's path test relies on (scratch writes only; the repository is only stat()ed):
O_RDONLY|O_TRUNC on this macOS, and whether the repository is reachable under spellings the hook's string-prefix
test does not match (a case variant on a case-insensitive volume; the /System/Volumes/Data firmlink)."""
import os
import sys
from pathlib import Path

work = Path(sys.argv[1]).resolve()
work.mkdir(parents=True, exist_ok=True)
p = work / "trunc_probe.txt"
p.write_text("hello")
fd = os.open(p, os.O_RDONLY | os.O_TRUNC)
os.close(fd)
print(f"O_RDONLY|O_TRUNC: opened; size after = {p.stat().st_size} (5 before)")
p.unlink()
repo = "/Users/gaborhorvath-ulenius/myprojects/neural-decompiler"
base = os.stat(repo)
for alt in (repo.replace("neural-decompiler", "NEURAL-DECOMPILER"), "/System/Volumes/Data" + repo):
    st = os.stat(alt)
    resolved = str(Path(alt).resolve())
    print(f"{alt}: same inode as the repository = {(st.st_dev, st.st_ino) == (base.st_dev, base.st_ino)}; "
          f"Path.resolve() = {resolved}; starts with the repository string = {resolved.startswith(repo)}")
