#!/usr/bin/env python3
import argparse
import os
import sqlite3
from contextlib import closing, contextmanager
from typing import List, Tuple, Dict, Set

@contextmanager
def connect(db_path: str):
    conn = sqlite3.connect(db_path)
    try:
        yield conn
    finally:
        conn.close()

def is_empty_db(conn: sqlite3.Connection) -> bool:
    cur = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type IN ('table','index','trigger','view')")
    return cur.fetchone()[0] == 0

def clone_schema_from_source(src_path: str, dest_conn: sqlite3.Connection, include_indexes_triggers: bool = True) -> None:
    with connect(src_path) as src:
        src.row_factory = sqlite3.Row
        cur = src.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master "
            "WHERE sql IS NOT NULL AND type IN ('table','index','trigger','view') "
            "ORDER BY CASE type WHEN 'table' THEN 0 WHEN 'index' THEN 1 WHEN 'trigger' THEN 2 ELSE 3 END"
        )
        for row in cur:
            t, name, tbl, sql = row["type"], row["name"], row["tbl_name"], row["sql"]
            if t == "table":
                # Avoid copying sqlite_sequence or internal tables
                if name.startswith("sqlite_"):
                    continue
                dest_conn.execute(sql)
            elif include_indexes_triggers:
                # Try to create indexes/triggers/views after tables exist
                try:
                    dest_conn.execute(sql)
                except sqlite3.DatabaseError:
                    # Some indexes might fail if columns differ; ignore silently
                    pass

def list_tables(conn: sqlite3.Connection) -> List[str]:
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")
    return [r[0] for r in cur.fetchall()]

def table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return [r[1] for r in cur.fetchall()]

def common_columns(src_conn: sqlite3.Connection, dest_conn: sqlite3.Connection, table: str) -> List[str]:
    sc = set(table_columns(src_conn, table))
    dc = set(table_columns(dest_conn, table))
    inter = [c for c in table_columns(dest_conn, table) if c in sc]  # keep dest order
    return inter

def copy_table(src_path: str, dest_conn: sqlite3.Connection, table: str, verbose: bool = True) -> Tuple[int, int]:
    """Copy data for a single table from src DB into dest connection using INSERT OR IGNORE.
    Returns:
        (inserted_rows, skipped_due_to_missing_table_or_columns)
    """
    inserted = 0
    skipped = 0
    with connect(src_path) as src_conn:
        src_tables = set(list_tables(src_conn))
        dest_tables = set(list_tables(dest_conn))
        if table not in src_tables:
            if verbose:
                print(f"[skip] {os.path.basename(src_path)} has no table '{table}'")
            return (0, 1)
        if table not in dest_tables:
            if verbose:
                print(f"[create] Creating missing table '{table}' from {os.path.basename(src_path)}")
            # Create the table definition only (no indexes/triggers here)
            # Extract CREATE TABLE sql for this table
            row = src_conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()
            if row and row[0]:
                dest_conn.execute(row[0])
            else:
                if verbose:
                    print(f"[warn] Could not fetch CREATE TABLE for '{table}' in {src_path}, skipping.")
                return (0, 1)

        cols = common_columns(src_conn, dest_conn, table)
        if not cols:
            if verbose:
                print(f"[skip] No common columns in table '{table}' between dest and {os.path.basename(src_path)}")
            return (0, 1)

        placeholders = ", ".join([f'"{c}"' for c in cols])
        sql = f'INSERT OR IGNORE INTO "{table}" ({placeholders}) SELECT {placeholders} FROM "{table}"'
        # Attach the source to the destination connection for fast copy
        alias = "src"
        dest_conn.execute(f"ATTACH DATABASE ? AS {alias}", (src_path,))
        try:
            cursor = dest_conn.execute(sql.replace(f'FROM "{table}"', f'FROM {alias}."{table}"'))
            inserted = dest_conn.execute("SELECT changes()").fetchone()[0]
            if verbose:
                print(f"[merge] {table}: +{inserted} rows from {os.path.basename(src_path)}")
        finally:
            dest_conn.execute(f"DETACH DATABASE {alias}")
    return (inserted, skipped)

def merge_sources(sources: List[str], dest: str, tables: List[str] = None, pragmas_fast: bool = True, verbose: bool = True) -> None:
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    with connect(dest) as dest_conn:
        if pragmas_fast:
            dest_conn.execute("PRAGMA journal_mode=WAL;")
            dest_conn.execute("PRAGMA synchronous=OFF;")
            dest_conn.execute("PRAGMA foreign_keys=OFF;")
            dest_conn.execute("PRAGMA temp_store=MEMORY;")

        # If destination is empty, clone schema from first source
        if is_empty_db(dest_conn):
            if verbose:
                print(f"[init] Destination DB is empty. Cloning schema from: {sources[0]}")
            clone_schema_from_source(sources[0], dest_conn, include_indexes_triggers=True)

        # Determine tables to process
        dest_tables = set(list_tables(dest_conn))
        if tables:
            target_tables = tables
        else:
            target_tables = sorted(dest_tables) if dest_tables else []
            # If dest has no tables (schema not cloned), use first source's tables
            if not target_tables:
                with connect(sources[0]) as s0:
                    target_tables = list_tables(s0)

        if verbose:
            print(f"[tables] Will merge tables: {', '.join(target_tables)}") 

        total_inserted = 0
        total_skipped = 0
        with dest_conn:
            for src in sources:
                if verbose:
                    print(f"[source] {src}")
                for t in target_tables:
                    ins, skip = copy_table(src, dest_conn, t, verbose=verbose)
                    total_inserted += ins
                    total_skipped += skip

        if verbose:
            print(f"[done] Inserted rows: {total_inserted}, Skips: {total_skipped}")
            # Optional vacuum to compact
            try:
                dest_conn.execute("VACUUM;")
            except sqlite3.DatabaseError:
                pass

def guess_langchain_tables(conn: sqlite3.Connection) -> List[str]:
    # Common LangChain SQLite cache tables (may vary by version/impl)
    candidates = [
        "cache",
        "cache_items",
        "embeddings",
        "fulltext_content",
        "fulltext_content_idx",
        "fulltext_content_data",
        "fulltext_content_docsize",
        "fulltext_content_config",
        # Add more if your cache uses additional tables
    ]
    existing = set(list_tables(conn))
    return [t for t in candidates if t in existing]

def main():
    parser = argparse.ArgumentParser(description="Merge multiple LangChain-style SQLite cache DBs into one.")
    parser.add_argument("dest", help="Destination SQLite DB path (will be created if not exists)")
    parser.add_argument("sources", nargs="+", help="Source SQLite DB files to merge")
    parser.add_argument("--tables", nargs="*", help="Specific tables to merge; default: all tables in dest (or first source if dest empty)")
    parser.add_argument("--langchain", action="store_true", help="Merge only common LangChain cache tables (auto-detected)")
    parser.add_argument("--no-fast-pragmas", action="store_true", help="Disable fast PRAGMAs (safer but slower)")
    parser.add_argument("-q", "--quiet", action="store_true", help="Less verbose output")
    args = parser.parse_args()

    # Basic validation
    missing = [s for s in args.sources if not os.path.exists(s)]
    if missing:
        raise SystemExit(f"Source files not found: {', '.join(missing)}")

    # If --langchain: detect tables from first source
    tables = args.tables
    if args.langchain and not tables:
        with connect(args.sources[0]) as s0:
            tables = guess_langchain_tables(s0)
            if not tables:
                print("[warn] Could not detect LangChain tables; will fall back to all tables.")
                tables = None

    merge_sources(
        sources=args.sources,
        dest=args.dest,
        tables=tables,
        pragmas_fast=not args.no_fast_pragmas,
        verbose=not args.quiet
    )

if __name__ == "__main__":
    main()