"""Auto-apply course image plotting style when repo root is on PYTHONPATH."""
try:
    from shared.plot_style import install
    install()
except Exception:
    # sitecustomize must never break unrelated Python commands.
    pass
