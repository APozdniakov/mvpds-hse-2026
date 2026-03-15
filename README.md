# mvpds-hse-2026

## Подготовка

```bash
pip install -U pip
pip install -r requirements.txt
ollama pull qwen3-embedding:4b
ollama pull qwen3:4b
```

### Первый запуск

Нужно указать pdf-документ, по которому должен строиться векторный индекс:

```bash
python3 -m src -i data_dir/input.csv --index-dir index_dir --from-input data_dir/input.pdf -o data_dir/output.csv
```

### Запуск

```bash
python3 -m src -i data_dir/input.csv --index-dir index_dir -o data_dir/output.csv
```
