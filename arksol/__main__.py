"""让 ``python -m arksol`` 可用。"""

from arksol.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
