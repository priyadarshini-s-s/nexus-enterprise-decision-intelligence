from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount


NEXUS_DATA = Mount(
    source=r"D:\Projects\NEXUS\data",
    target="/opt/nexus/data",
    type="bind",
)

NEXUS_DOCS = Mount(
    source=r"D:\Projects\NEXUS\docs",
    target="/opt/nexus/docs",
    type="bind",
)

NEXUS_REPORTS = Mount(
    source=r"D:\Projects\NEXUS\reports",
    target="/opt/nexus/reports",
    type="bind",
)


default_args = {
    "owner": "nexus-data-platform",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}


with DAG(
    dag_id="nexus_data_pipeline",
    description="NEXUS production data quality and dbt pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    default_args=default_args,
    dagrun_timeout=timedelta(minutes=30),
    tags=["nexus", "data-platform", "dq", "dbt"],
) as dag:

    validate_silver = DockerOperator(
        task_id="validate_silver",
        image="nexus-dq-spark:1.0",
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        auto_remove="success",
        mount_tmp_dir=False,
        mounts=[NEXUS_DATA],
        command=[
            "python3",
            "-c",
            (
                "from pyspark.sql import SparkSession; "
                "p='/opt/nexus/data/silver/olist_spark/orders'; "
                "s=SparkSession.builder.master('local[2]').appName('NEXUS-Silver-Validation').getOrCreate(); "
                "df=s.read.parquet(p); "
                "n=df.count(); "
                "print(f'SILVER ROW COUNT: {n}'); "
                "assert n == 99441, f'Expected 99441 rows, got {n}'; "
                "print('SILVER VALIDATION: PASS'); "
                "s.stop()"
            ),
        ],
    )


    dq_gate = DockerOperator(
        task_id="dq_gate",
        image="nexus-dq-spark:1.0",
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        auto_remove="success",
        mount_tmp_dir=False,
        mounts=[NEXUS_DATA, NEXUS_DOCS, NEXUS_REPORTS],
        command=[
            "python3",
            "-m",
            "scripts.data_quality.run_dq",
            "--contract",
            "/opt/nexus/docs/data_contracts/olist_orders_contract.yml",
            "--input",
            "/opt/nexus/data/silver/olist_spark/orders",
            "--format",
            "parquet",
            "--report",
            "/opt/nexus/reports/data_quality/dq_report.json",
        ],
        environment={
            "PYTHONPATH": "/opt/nexus",
        },
    )


    dbt_build = DockerOperator(
        task_id="dbt_build",
        image="nexus-data-platform:1.5",
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        auto_remove="success",
        mount_tmp_dir=False,
        mounts=[NEXUS_DATA],
        command=[
            "dbt",
            "build",
            "--project-dir",
            "/opt/nexus/nexus_dbt",
            "--profiles-dir",
            "/opt/nexus/.dbt",
            "--vars",
            '{"nexus_data_root": "/opt/nexus/data"}',
        ],
        working_dir="/opt/nexus/nexus_dbt",
    )


    dbt_test = DockerOperator(
        task_id="dbt_test",
        image="nexus-data-platform:1.5",
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        auto_remove="success",
        mount_tmp_dir=False,
        mounts=[NEXUS_DATA],
        command=[
            "dbt",
            "test",
            "--project-dir",
            "/opt/nexus/nexus_dbt",
            "--profiles-dir",
            "/opt/nexus/.dbt",
            "--vars",
            '{"nexus_data_root": "/opt/nexus/data"}',
        ],
        working_dir="/opt/nexus/nexus_dbt",
    )


    gold_validation = DockerOperator(
        task_id="gold_validation",
        image="nexus-data-platform:1.5",
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        auto_remove="success",
        mount_tmp_dir=False,
        mounts=[NEXUS_DATA],
        command=[
            "python",
            "-c",
            (
                "import duckdb; "
                "p='/opt/nexus/data/warehouse/nexus.duckdb'; "
                "con=duckdb.connect(p); "
                "tables=['dim_customer_360','fct_category_kpis',"
                "'fct_customer_cohort_retention','fct_executive_kpis',"
                "'fct_olist_order_performance','fct_product_kpis']; "
                "existing={r[0] for r in con.execute('SHOW TABLES').fetchall()}; "
                "missing=set(tables)-existing; "
                "assert not missing, f'Missing gold objects: {missing}'; "
                "[(__import__('builtins').exec("
                "f\"assert {con.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]} > 0, '{t} is empty'\""
                ")) for t in tables]; "
                "print('GOLD VALIDATION: PASS'); "
                "print('Validated:', ', '.join(tables)); "
                "con.close()"
            ),
        ],
        working_dir="/opt/nexus",
    )


    validate_silver >> dq_gate >> dbt_build >> dbt_test >> gold_validation
