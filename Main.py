import duckdb
from splink import Linker #Linker, Splink’in ana sınıfıdır.
from splink.internals.duckdb.database_api import DuckDBAPI
from splink.datasets import splink_datasets
from splink.comparison_library import NameComparison, DateOfBirthComparison
from splink.blocking_rule_library import block_on

connection = duckdb.connect()
df = splink_datasets.historical_50k
connection.register("historical_50k", df)#df’yi(Pandas DataFrame) DuckDB'ye kaydettik.

settings = {
    "link_type": "dedupe_only",#Tekrar eden kayıtlar eşleştirilecek
    "comparisons": [
        NameComparison("first_name"),
        NameComparison("surname"),
        DateOfBirthComparison("dob", input_is_string=True)
    ],
    "blocking_rules_to_generate_predictions": [#bloklama kuralları.
        block_on("first_name", "surname")#Sadece aynı isim ve soyisim grubundakiler eşleşme için karşılaştırılır.
    ],
    "retain_matching_columns": True,
    "retain_intermediate_calculation_columns": True# Sonuç DataFrame’inde karşılaştırma sütunları ve ara hesaplamalar tutulur.
}

db_api = DuckDBAPI(connection)#Veritabanı arayüzü oluşturduk

linker = Linker(#settings ve db_api ile eşleşme işlemlerini yapacak.
    input_table_or_tables="historical_50k",
    settings=settings,
    db_api=db_api
)

linker.training.estimate_u_using_random_sampling(max_pairs=1e7)#U olasılıkları, farklı kayıtların alan bazında benzerlik göstermeme
# ihtimalini tahmin ederek modelin doğru eşleşmeleri öğrenmesini sağlar.

em_training_session = linker.training.estimate_parameters_using_expectation_maximisation(#Verideki gizli parametreleri tahmin etmek için kullanılan istatistiksel bir yöntem.
    blocking_rule=block_on("dob"),#dob üzerinden bloklama yapılarak m ve u olasılıkları gibi parametreler güncelleniyor.
    fix_u_probabilities=False
)

linker.training.estimate_probability_two_random_records_match(#İki rastgele kaydın aynı kişi olma olasılığı tahmin ediliyor.
    deterministic_matching_rules=[block_on("dob")],#dob alanında bloklama kullanılıyor.
    recall=0.8#%80 duyarlılık hedefleniyor.
)

result = linker.inference.predict()#Kayıtların eşleşme ihtimalleri tahmin ediliyor.

df_predict = result.as_pandas_dataframe()#Pandas DataFrame formatına dönüştürülüyor.

conn_results = duckdb.connect("splink_results.duckdb")
conn_results.register("df_result", df_predict)#Pandas DataFrame veritabanına kayıt ediliyor
conn_results.execute("DROP TABLE IF EXISTS matched_records")
conn_results.execute("CREATE TABLE matched_records AS SELECT * FROM df_result")
df = conn_results.execute('SELECT * FROM matched_records LIMIT 10').fetchdf()
print(df)
conn_results.close()
