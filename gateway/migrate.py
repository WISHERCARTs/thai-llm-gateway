"""Apply db/migrations/*.sql in name order, once each: `python -m gateway.migrate`.

Uses DATABASE_URL. Applied files are recorded in public.schema_migrations.
"""

import asyncio
import os
import sys
from pathlib import Path

MIGRATIONS = Path(__file__).resolve().parents[1] / "db" / "migrations"


async def migrate(dsn: str, folder: Path = MIGRATIONS) -> list[str]:
    import asyncpg

    conn = await asyncpg.connect(dsn, statement_cache_size=0)
    try:
        await conn.execute(
            "create table if not exists public.schema_migrations "
            "(name text primary key, applied_at timestamptz not null default now())"
        )
        done = {r["name"] for r in await conn.fetch("select name from public.schema_migrations")}
        applied = []
        for path in sorted(folder.glob("*.sql")):
            if path.name in done:
                continue
            async with conn.transaction():
                await conn.execute(path.read_text(encoding="utf-8"))
                await conn.execute("insert into public.schema_migrations (name) values ($1)", path.name)
            applied.append(path.name)
        return applied
    finally:
        await conn.close()


def main() -> int:
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        print("DATABASE_URL is not set", file=sys.stderr)
        return 1
    for name in asyncio.run(migrate(dsn)):
        print("applied", name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
