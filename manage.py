
#!/usr/bin/env python
"""
Django's command-line utility for administrative tasks.
"""
import os
import sys

# --- Python 3.13 dataclasses shim (restores _create_fn for Strawberry) ----------
import dataclasses
if not hasattr(dataclasses, "_create_fn"):
    def _create_fn(name, args, body, *, globals=None, locals=None, return_type=None):
        src = f"def {name}({args}):\n" + "\n".join(f"    {line}" for line in body)
        ns: dict[str, object] = {}
        exec(src, globals if globals is not None else {}, ns)
        fn = ns[name]
        if return_type is not None:
            fn.__annotations__["return"] = return_type
        return fn
    dataclasses._create_fn = _create_fn
# ------------------------------------------------------------------------------

def main():
    """Run administrative tasks."""
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "pla_sim.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)

if __name__ == "__main__":
    main()

