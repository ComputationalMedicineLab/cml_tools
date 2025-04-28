"""Common patterns using databricks-sql-connector"""
import contextlib
import datetime
import decimal
import os
import pickle

from databricks import sql as dbsql
import pyarrow.parquet as pq


def _validate_connection_opts(opts: dict):
    # Quick and dirty argument validation
    assert len(opts) == 3, opts.keys()
    for key in ('server_hostname', 'http_path', 'access_token'):
        assert isinstance(opts.get(key), str)


def select_arrow(select_statement: str,
                 conn_opts: dict[str, str],
                 arraysize: int = 100_000,
                 combine_chunks: bool = True,
                 pq_file: str = None):
    """
    Run a SELECT statement and greedily `fetchall_arrow()` the results. The
    arrow format is significantly more memory efficient than the sql
    connector's Row class, so this often is the fastest and most efficient way
    to pull very large datasets from databricks.
    """
    _validate_connection_opts(conn_opts)

    with dbsql.connect(**conn_opts) as connection:
        with connection.cursor(arraysize=arraysize) as cursor:
            cursor.execute(select_statement)
            results = cursor.fetchall_arrow()

    if combine_chunks:
        results = results.combine_chunks()

    if isinstance(pq_file, str):
        pq.write_table(results, pq_file)

    return results


def select_python(select_statement: str,
                  conn_opts: dict[str, str],
                  arraysize: int = 100_000,
                  include_header: bool = True,
                  convert_tuples: bool = True,
                  outfile: str|os.PathLike = None):
    """
    Run a SELECT statement and greedily `fetchall()` the results. Not suitable
    for very large result sets; use `select_arrow` or stream the results with a
    loop over `fetchmany()`.
    """
    _validate_connection_opts(conn_opts)
    with dbsql.connect(**conn_opts) as connection:
        with connection.cursor(arraysize=arraysize) as cursor:
            cursor.execute(select_statement)
            header = tuple(name for (name, *_) in cursor.description)
            result = cursor.fetchall()
    if include_header:
        result.insert(0, header)
    if convert_tuples:
        result = tuple(tuple(r) for r in result)
    if outfile is not None:
        with open(outfile, 'wb') as file:
            pickle.dump(result, file, protocol=pickle.HIGHEST_PROTOCOL)
    return header, result


def split_statements(statements: str|list[str]):
    """
    The databricks cursor.execute function does not support multiple SQL
    statements per input. However, to keep the source scripts readable it is
    desirable to allow multiple statements per input file. This function strips
    out all comments, extra space, and splits its inputs by ";" chars.

    Argument `statements` may be a single string or an iterable of strings;
    each string may contain 1 or more SQl statements.
    """
    if isinstance(statements, str):
        statements = [statements]
    scripts = []
    for stmt in statements:
        buff = []
        for line in stmt.strip().split('\n'):
            if '--' in line:
                # If the line ends in a comment this removes it; if the whole
                # line is a comment this reduces it to the empty string ''
                line = line[:line.index('--')].rstrip()
            if line:
                buff.append(line)
                if line.rstrip().endswith(';'):
                    scripts.append('\n'.join(buff))
                    buff = []
    return tuple(scripts)


def run_scripts(statement: str|list[str],
                conn_opts: dict[str, str]):
    """Execute the non-SELECT statement(s) and commit"""
    _validate_connection_opts(conn_opts)
    scripts = split_statements(statement)
    # Transactions are not supported by databricks, so we don't need to bother
    # with committing / rollbacking etc - we can just execute each statement in
    # sequence and it should work correctly. `cursor.execute` returns None.
    with dbsql.connect(**conn_opts) as connection:
        with connection.cursor() as cursor:
            for script in scripts:
                cursor.execute(script)
