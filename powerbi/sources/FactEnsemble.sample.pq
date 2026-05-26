// Sample data (offline / template). Set SampleDataPath parameter to the CSV location.
let
    Source = Csv.Document(
        File.Contents(SampleDataPath),
        [Delimiter = ",", Columns = 4, Encoding = 65001, QuoteStyle = QuoteStyle.None]
    ),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),
    Typed = Table.TransformColumnTypes(
        Promoted,
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
