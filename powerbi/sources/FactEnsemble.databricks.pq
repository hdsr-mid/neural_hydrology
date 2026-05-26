// Production: Unity Catalog table dbw_datascience_tst_weu_001.default.output_forecast
// In Power BI: Gegevens ophalen → Databricks → selecteer de tabel, of plak deze query na parameterisatie.
let
    Source = Databricks.Catalogs(
        "adb-3159846276042543.3.azuredatabricks.net",
        "dbw_datascience_tst_weu_001",
        [Catalog = "dbw_datascience_tst_weu_001", Database = "default", EnableAutomaticProxyDiscovery = true]
    ),
    Table = Source{[Schema = "default", Item = "output_forecast", Kind = "Table"]}[Data],
    Typed = Table.TransformColumnTypes(
        Table,
        {
            {"datetime", type datetime},
            {"ensemble_id", Int64.Type},
            {"afvoergeb_id", type text},
            {"value", type number}
        }
    ),
    Sorted = Table.Sort(Typed, {{"afvoergeb_id", Order.Ascending}, {"ensemble_id", Order.Ascending}, {"datetime", Order.Ascending}})
in
    Sorted
