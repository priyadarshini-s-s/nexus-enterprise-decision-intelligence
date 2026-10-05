{% macro iceberg_source(path) %}

    iceberg_scan('{{ path }}')

{% endmacro %}