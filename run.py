import sys

sys.path.append("./src")
try:
    import ssmerge46

    ssmerge46.start()
except Exception as e:
    print(f"Unknown error occured : {e}", file=sys.stderr)
    sys.exit(1)
