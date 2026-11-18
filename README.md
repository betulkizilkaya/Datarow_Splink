# Datarow_Splink – Entity Resolution and Record Deduplication

A Python-based entity resolution project that uses Splink and DuckDB to identify potentially duplicated records in a structured dataset.

## Project Overview

The project implements a probabilistic record linkage workflow using Splink.

It uses Splink's `historical_50k` sample dataset and applies field-level comparisons to identify records that may refer to the same person.

The matching process considers first names, surnames, and dates of birth. DuckDB is used as the database backend for record linkage and for storing the generated matching results.

## Features

- **Dataset Integration:** Uses Splink's built-in `historical_50k` sample dataset.
- **Record Deduplication:** Configures Splink for deduplication within a single dataset.
- **Field-Level Comparisons:** Compares first names, surnames, and dates of birth.
- **Blocking:** Uses first name and surname to generate candidate record pairs for prediction.
- **Parameter Estimation:** Applies random sampling and Expectation-Maximization to estimate model parameters.
- **Probabilistic Matching:** Generates match predictions using Splink's inference workflow.
- **CSV Export:** Exports predicted matching records to a CSV file.
- **DuckDB Storage:** Stores matching results in a DuckDB database.

## Technologies Used

- Python
- Splink
- DuckDB
- Pandas

## Project Structure

```text
Datarow_Splink/
├── Main.py
├── matched_records.csv
├── splink_results.duckdb
├── README.md
└── LICENSE
```

- `Main.py`: Configures and executes the record linkage workflow.
- `matched_records.csv`: Exported matching predictions.
- `splink_results.duckdb`: DuckDB database containing matching results.

## How It Works

### 1. Dataset Loading

The project loads the `historical_50k` dataset from Splink:

```python
from splink.datasets import splink_datasets

df = splink_datasets.historical_50k
```

The dataset is registered in DuckDB for processing.

### 2. Linkage Configuration

The project configures Splink for deduplication within one dataset (`dedupe_only`). Comparisons are defined for:

- First name
- Surname
- Date of birth

The prediction blocking rule groups records based on first name and surname.

### 3. Candidate Generation

Blocking reduces the number of record pairs considered during prediction. The configured prediction blocking rule uses first name and surname.

### 4. Model Parameter Estimation

The script performs parameter estimation steps, including:

- Estimating u-probabilities using random sampling with a configured `max_pairs` value of `1e7`.
- Using Expectation-Maximization with a date-of-birth blocking rule.
- Estimating match probabilities for record pairs.

The script configures `recall=0.8` in the probability estimation step.

**This is a configured assumption/target used by the estimation procedure, not a measured recall score or an independently verified model result.**

### 5. Match Prediction

Splink generates probabilistic matching predictions through its inference workflow. The resulting predictions are converted into a Pandas DataFrame.

### 6. Result Export

The matching predictions are exported to:

```text
matched_records.csv
```

The results are also written to the DuckDB database:

```text
splink_results.duckdb
```

The output table is named `matched_records`.

## Installation

Clone the repository:

```bash
git clone https://github.com/betulkizilkaya/Datarow_Splink.git
cd Datarow_Splink
```

Install the required packages:

```bash
pip install splink duckdb pandas
```

The project uses Splink's DuckDB backend. Use a Splink version compatible with the API imports and configuration in `Main.py`.

## Usage

Run the main script:

```bash
python Main.py
```

The script loads the sample dataset, estimates matching parameters, generates record linkage predictions, and saves the results as CSV and DuckDB outputs.

## Output Files

| File | Description |
|---|---|
| `matched_records.csv` | Exported record matching predictions |
| `splink_results.duckdb` | DuckDB database containing the `matched_records` table |

## Important Notes

- The project uses Splink's built-in `historical_50k` sample dataset.
- The configured `recall=0.8` value should not be interpreted as achieved recall.
- The repository does not include a ground-truth evaluation demonstrating precision, recall, or F1-score.
- Predicted pairs are probabilistic candidate matches; they are not independently verified as true duplicates.
- The script prints a sample of the stored matching results.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
