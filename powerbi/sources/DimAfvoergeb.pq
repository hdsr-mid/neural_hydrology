let
    Source = FactEnsemble,
    Distinct = Table.Distinct(Table.SelectColumns(Source, {"afvoergeb_id"})),
    Sorted = Table.Sort(Distinct, {{"afvoergeb_id", Order.Ascending}})
in
    Sorted
