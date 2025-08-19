import numpy as np
import pandas
import pyarrow as pa
import tlc
from tqdm import tqdm

n_rows = 477715
embedding_dim = 1280


def peek_parquet_file(file_path: str):
    df = pandas.read_parquet(file_path)
    print(df.head())
    print(len(df["embeddings"][0]))


def batched_data_generator(batch_size: int = 32):
    """Returns batches of data like {"col_name": [batch_data], ..}"""

    # Generate data:
    example_ids = np.arange(n_rows)
    losses = np.random.rand(n_rows)
    predicted = np.random.randint(0, 1, n_rows)
    confidence = np.random.rand(n_rows)
    top1_accuracy = np.random.rand(n_rows)
    embeddings = np.random.rand(n_rows, embedding_dim)
    epoch = np.zeros(n_rows)
    training_phase = np.zeros(n_rows)
    input_table_id = np.zeros(n_rows)

    # Create batches:
    for i in tqdm(range(0, n_rows, batch_size), desc="Generating batches", total=n_rows // batch_size):
        yield {
            "example_ids": example_ids[i : i + batch_size],
            "losses": losses[i : i + batch_size],
            "predicted": predicted[i : i + batch_size],
            "confidence": confidence[i : i + batch_size],
            "top1_accuracy": top1_accuracy[i : i + batch_size],
            "embeddings": embeddings[i : i + batch_size],
            "epoch": epoch[i : i + batch_size],
            "training_phase": training_phase[i : i + batch_size],
            "input_table_id": input_table_id[i : i + batch_size],
        }


def write_table():
    data_generator = batched_data_generator(batch_size=32)
    table_writer = tlc.TableWriter(
        project_name="SPOOR-BIRD-CLS",
        table_name="debug_oom_issues",
    )
    for batch in data_generator:
        table_writer.add_batch(batch)

    table = table_writer.finalize()
    print(f"Wrote {table.get_row_cache_size()} bytes")
    tlc.ObjectRegistry._delete_object_from_caches(table.url)


if __name__ == "__main__":
    for epoch in range(10):
        print(f"Epoch {epoch}")
        print(f"Pyarrow memory before writing: {pa.total_allocated_bytes()}")
        write_table()
        print(f"Pyarrow memory after writing: {pa.total_allocated_bytes()}")
