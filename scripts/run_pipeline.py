#!/usr/bin/env python
import subprocess, sys, pathlib

here = pathlib.Path(__file__).resolve().parent
script = here / "konya_gw_all_in_one_EN.py"
ret = subprocess.call([sys.executable, str(script)] + sys.argv[1:])
sys.exit(ret)
