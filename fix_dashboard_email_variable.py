# -*- coding: utf-8 -*-
"""Dashboard NameError-only repair. Does not change CSS/layout/features."""
from pathlib import Path
path = Path(__file__).resolve().parent / "stock_dashboard.py"
s = path.read_text(encoding="utf-8")
if "if email_send_latest:" not in s:
    print("OK: no undefined email_send_latest usage found.")
    raise SystemExit(0)
if "email_send_latest = False" in s:
    print("OK: email_send_latest already initialized; no change.")
    raise SystemExit(0)
pos = s.find("if email_send_latest:")
line_start = s.rfind("\n", 0, pos) + 1
indent = "    " if s[line_start:pos].strip() == "" else ""
# Use the function body indentation around the failing line.
line = s[line_start:pos]
indent = line[:len(line)-len(line.lstrip())]
s = s[:line_start] + indent + "email_send_latest = False\n" + indent + "email_force = False\n" + s[line_start:]
path.write_text(s, encoding="utf-8")
print("OK: only initialized email_send_latest/email_force before first use. No layout changes.")
