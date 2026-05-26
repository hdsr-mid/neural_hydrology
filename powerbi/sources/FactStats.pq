// Percentiles per datetime and afvoergeb_id (matches plot_ensemble_forecast.py).
let
    Source = FactEnsemble,
    Grouped = Table.Group(
        Source,
        {"datetime", "afvoergeb_id"},
        {
            {
                "Stats",
                (t as table) =>
                    let
                        vals = List.RemoveNulls(t[value])
                    in
                        [
                            p05 = List.Percentile(vals, 0.05),
                            p25 = List.Percentile(vals, 0.25),
                            median = List.Percentile(vals, 0.5),
                            p75 = List.Percentile(vals, 0.75),
                            p95 = List.Percentile(vals, 0.95)
                        ],
                type record
            }
        }
    ),
    Expanded = Table.ExpandRecordColumn(Grouped, "Stats", {"p05", "p25", "median", "p75", "p95"}),
    Typed = Table.TransformColumnTypes(
        Expanded,
        {
            {"datetime", type datetime},
            {"afvoergeb_id", type text},
            {"p05", type number},
            {"p25", type number},
            {"median", type number},
            {"p75", type number},
            {"p95", type number}
        }
    )
in
    Typed
