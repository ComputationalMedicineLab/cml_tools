"""Common patterns using databricks-sql-connector"""
import contextlib
import datetime
import decimal
import os
import pickle

from databricks import sql as dbsql


def convert(x):
    """Convert or modify types as needed, else identity"""
    match type(x):
        # the sql lib will spuriously attach a timezone
        case datetime.datetime: return x.replace(tzinfo=None)
        # We usually want to operate on basic floats
        case decimal.Decimal: return float(x)
        case _: return x


def row_factory(row, convert=convert):
    """Convert the elements of `row` using function `convert`"""
    return tuple(convert(x) for x in row)


def select(cursor, sql: str, batchsize=4096, row_factory=row_factory):
    """Generator through the result set of a sql SELECT query"""
    cursor.execute(sql)
    tuples = cursor.fetchmany(batchsize)
    while tuples:
        yield from (row_factory(r) for r in tuples)
        tuples = cursor.fetchmany(batchsize)


def _validate_conn_opt(opts: dict):
    assert 'server_hostname' in opts
    assert 'http_path' in opts
    assert 'access_token' in opts


def select_ctx(sql: str, conn_opt: dict, select_opt: dict = None):
    """Yield SELECT results with connection and cursor context"""
    _validate_conn_opt(conn_opt)
    if select_opt is None: select_opt = {}
    with dbsql.connect(**conn_opt) as connection:
        with connection.cursor() as cursor:
            yield from select(cursor, sql, **select_opt)


def pickle_select(outfile: str|os.PathLike,
                  sql: str,
                  conn_opt: dict[str, str],
                  select_opt: dict[str, object] = None,
                  header: object = None,
                  consume: bool = True):
    """
    Arguments
    ---------
    outfile: str|os.PathLike
        The output file into which to pickle the results.
    sql: str
        The SQL containing the select statement to run.
    conn_opt: dict[str, str]
        A dictionary with keyword args for the databricks.sql.connect function
    select_opt: dict[str, object]
        A dictionary with keyword arguments for the function `select`
    header: object
        Any object to serialize to the outputs before the SQL result set
    consume: bool
        If True, read all the results at once and serialize as a single list.
        Else serialize each result as it comes (the output file is then a
        pickle stream, rather than a single serialized object).
    """
    _validate_conn_opt(conn_opt)

    results = []
    if header is not None:
        results.append(header)

    if consume:
        results.extend(select_ctx(sql, conn_opt, select_opt))
        with open(outfile, 'wb') as file:
            pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(outfile, 'wb') as file:
            for row in results:
                pickle.dump(row, file, protocol=pickle.HIGHEST_PROTOCOL)
            for row in select_ctx(sql, conn_opt, select_opt):
                pickle.dump(row, file, protocol=pickle.HIGHEST_PROTOCOL)
