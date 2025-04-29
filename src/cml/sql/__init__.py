import importlib.resources as ipr
import textwrap as tw

# This is how to get access to the sql templates in this subpackage directory
# see: https://docs.python.org/3/library/importlib.resources.html
# and: https://importlib-resources.readthedocs.io/en/latest/using.html
def get_template(name, strip_comments=True):
    """Load a SQL template from the cml.sql resource directory"""
    base = ipr.files('cml.sql').joinpath(name).read_text().strip()
    if strip_comments:
        base = (x for x in base.split('\n') if not x.lstrip().startswith('--'))
        base = '\n'.join(base)
    return base


def indent(s):
    """Helper function. Indent each line of a string by 4 space chars"""
    return tw.indent(s, '    ')


class DataTableParams:
    """
    Container for parameters common to the SQL needed to generate source data
    tables, counts over those tables, and metadata about the channels in those
    tables.
    """
    def __init__(self, name, metatable, fill_expr, mode_expr, where, value):
        self.name = name
        self.metatable = metatable
        self.fill_expr = fill_expr
        self.mode_expr = mode_expr
        self.where = where
        self.value = value

# The most common parameters we use for four major data types:
ConditionParams = DataTableParams(
    name='conditions',
    metatable='conditions',
    fill_expr='0.00013689253935660506',
    mode_expr="'Condition'",
    where='mentions >= 1000',
    value=False,
)

MeasurementParams = DataTableParams(
    name='measurements',
    metatable='measurement_stats',
    fill_expr='p50',
    mode_expr="'Measurement'",
    where='mentions >= 1000 AND persons >= 100',
    value=True,
)

MedicationParams = DataTableParams(
    name='medications',
    metatable='medications',
    fill_expr='0.0',
    mode_expr="'Medication'",
    where='mentions >= 1000',
    value=False,
)

ProcedureParams = DataTableParams(
    name='procedures',
    metatable='procedures',
    fill_expr='0.0',
    mode_expr="'Procedure'",
    where='mentions >= 100',
    value=False,
)


class SQLGenerator:
    def __init__(self,
                 prefix='',
                 concept_stem='concepts',
                 cohort_stem='persons',
                 birthdate='1920-01-01',
                 startdate='2000-01-01',
                 data_spec=(ConditionParams, MeasurementParams,
                            MedicationParams, ProcedureParams),
                 source_catalog='victr_sd',
                 source_schema='sd_omop_prod',
                 wspace_catalog='workspace_victrsd',
                 wspace_schema='lasko_ecd',
                 number=False,
                 ):
        self.prefix = prefix
        self.concept_stem = concept_stem
        self.cohort_stem = cohort_stem
        self.birthdate = birthdate
        self.startdate = startdate
        self.data_spec = data_spec
        self.source_catalog = source_catalog
        self.source_schema = source_schema
        self.wspace_catalog = wspace_catalog
        self.wspace_schema = wspace_schema
        # TODO: write outputs to file. if number, number the files.
        self.number = number

    @property
    def concept_table(self):
        if self.prefix:
            return f'{self.prefix}_{self.concept_stem}'
        return self.concept_stem

    @property
    def cohort_table(self):
        if self.prefix:
            return f'{self.prefix}_{self.cohort_stem}'
        return self.cohort_stem

    def format(self, s, **kwargs):
        fmtkws = {'workspace': self.wspace_catalog,
                  'schema': self.wspace_schema,
                  'omop': self.source_schema,
                  'source': self.source_catalog,
                  'birthdate': self.birthdate,
                  'startdate': self.startdate,
                  'concept_table': self.concept_table,
                  'cohort_table': self.cohort_table}
        fmtkws.update(kwargs)
        return s.format(**fmtkws)

    def union(self, parts):
        return '\n    UNION\n'.join(parts)

    def gen_create_as(self, select, table):
        """Produce CREATE AS statement for `table` from `select`"""
        select = indent(self.format(select))
        return self.format(tw.dedent("""
            DROP TABLE IF EXISTS {workspace}.{schema}.{table};

            CREATE TABLE {workspace}.{schema}.{table} AS (
            {select}
            );
        """).strip(), select=select, table=table)

    def gen_templated_table(self, table):
        """Produce the CREATE SQL for table from a SELECT template"""
        return self.gen_create_as(get_template(f'{table}.sql'), table)

    def gen_counts(self, tablename):
        """Produce SQL to SELECT counts over data table {tablename}"""
        template = get_template('counts.sql', strip_comments=True)
        return self.format(template, tablename=tablename)

    def gen_counts_table(self, tablename):
        """Produce SQL to CREATE a {tablename}_counts table"""
        select = self.gen_counts(tablename)
        return self.gen_create_as(select, f'{tablename}_counts')

    def gen_concept_table(self, create=True):
        """Produce SQL to SELECT (or CREATE) the set of viable concept_ids"""
        template = tw.dedent("""
            SELECT concept_id
            FROM {workspace}.{schema}.{name}_counts
            WHERE {where}
        """).strip()
        select = self.union(
            self.format(template, name=spec.name, where=spec.where)
            for spec in self.data_spec
        )
        if create:
            return self.gen_create_as(select, self.concept_table)
        return select

    def gen_cohort_table(self, create=True, intersect_on=None):
        """Produce SQL to SELECT (or CREATE) the set of viable person_ids"""
        template = get_template('cohort.sql', strip_comments=True)
        select = self.union(
            self.format(template, name=spec.name)
            for spec in self.data_spec
        )
        if intersect_on is not None:
            select = indent(select)
            filter = self.format(intersect_on)
            select = f'(\n{select}\n)\nINTERSECT\n{filter}'
        if create:
            return self.gen_create_as(select, self.cohort_table)
        return select

    def gen_select_data(self, date_convert=False):
        """Produce a SELECT statement to get the main EHR data"""
        select = get_template('select_data.sql')
        subsel = get_template('select_data_cte.sql')
        parts = []
        for spec in self.data_spec:
            if spec.value:
                value_expr = 'value'
            else:
                value_expr = 'NULL AS value'
            s = self.format(subsel, name=spec.name, value_expr=value_expr)
            parts.append(s)
        cte = indent(self.union(parts))
        if date_convert:
            date_expr = 'datetime::DATE'
        else:
            date_expr = 'datetime::TIMESTAMP_NTZ'
        return self.format(select, cte=cte, date_expr=date_expr)

    def gen_select_concept_meta(self):
        """Produce a SELECT to get concept metadata from OMOP concept table"""
        select = get_template('select_meta.sql')
        subsel = get_template('select_meta_cte.sql')
        cte = indent(self.union(
            self.format(subsel, metatable=spec.metatable,
                        fill_expr=spec.fill_expr,
                        mode_expr=spec.mode_expr)
            for spec in self.data_spec
        ))
        return self.format(select, cte=cte)

    def gen_select_demographics(self):
        """Produce a SELECT to get person age and demographics data"""
        return self.format(get_template('select_demo.sql'))

    def gen_select_demographics_meta(self):
        """Produce a SELECT to get demographics channel metadata"""
        return self.format(get_template('select_demo_meta.sql'))

    def gen_select_visits(self):
        """Produce a SELECT to get visit data from the workspace"""
        select = get_template('select_visits.sql')
        subsel = get_template('select_visits_cte.sql')
        cte = indent(indent(self.union(
            self.format(subsel, name=spec.name)
            for spec in self.data_spec
        )))
        return self.format(select, cte=cte)
