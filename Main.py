import duckdb
from splink import Linker
from splink.internals.duckdb.database_api import DuckDBAPI
from splink.datasets import splink_datasets
from splink.comparison_library import NameComparison, DateOfBirthComparison
from splink.blocking_rule_library import block_on

# 1. DuckDB bağlantısı ve veri yükleme
connection = duckdb.connect()
df = splink_datasets.historical_50k
connection.register("historical_50k", df)

# 2. Ayarları tanımla (comparison ve blocking kuralları)
settings = {
    "link_type": "dedupe_only",
    "comparisons": [
        NameComparison("first_name"),
        NameComparison("surname"),
        DateOfBirthComparison("dob", input_is_string=True)
    ],
    "blocking_rules_to_generate_predictions": [
        block_on("first_name", "surname")
    ],
    "retain_matching_columns": True,
    "retain_intermediate_calculation_columns": True
}

# 3. DuckDBAPI nesnesi oluştur
db_api = DuckDBAPI(connection)

# 4. Linker nesnesi oluştur
linker = Linker(
    input_table_or_tables="historical_50k",
    settings=settings,
    db_api=db_api
)

# 5. U olasılıklarını tahmin et (örnekleme ile)
linker.training.estimate_u_using_random_sampling(max_pairs=1e7)

# 6. EM eğitimi başlat (dob ile bloklama yapılacak)
em_training_session = linker.training.estimate_parameters_using_expectation_maximisation(
    blocking_rule=block_on("dob"),
    fix_u_probabilities=False
)

# 7. Random eşleşme olasılığını tahmin et
linker.training.estimate_probability_two_random_records_match(
    deterministic_matching_rules=[block_on("dob")],
    recall=0.8
)

# 8. Eşleşmeleri tahmin et
result = linker.inference.predict()

# 9. Sonuçları Pandas DataFrame'e çevir
df_predict = result.as_pandas_dataframe()

# 10. Sonuçları DuckDB veritabanına kaydet
conn_results = duckdb.connect("splink_results.duckdb")
conn_results.register("df_result", df_predict)
conn_results.execute("DROP TABLE IF EXISTS matched_records")
conn_results.execute("CREATE TABLE matched_records AS SELECT * FROM df_result")
conn_results.close()
