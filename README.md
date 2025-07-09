# Language Identifier CLI

A CLI and Python library for detecting the language of text using n-gram frequency analysis (Cavnar & Trenkle).

## CLI Usage

```bash
python cli.py [-h] {train,identify,info} ...
```

### Commands

1. train
   Train a new model. By default it:
   - Uses samples in `dataset/` (omit `-d`)
   - Saves model as `test_model.lid` in `models/` (omit `-o`)
   ```bash
   # Train with defaults (dataset/ → models/test_model.lid)
   python cli.py train

   # Or override output model name and options:
   python cli.py train -o my_model [--min-ngram N] [--max-ngram N] [--profile-size N]
   ```
   Arguments:
   - `-d, --dir DIR`           Directory containing language samples (defaults to `dataset/`)
   - `-o, --output OUTPUT`     Base name for the output model file (defaults to `test_model`, saved as OUTPUT.lid)
   - `--min-ngram N`           Minimum n-gram size (default: `1`)
   - `--max-ngram N`           Maximum n-gram size (default: `5`)
   - `--profile-size N`        Number of n-grams in each language profile (default: `300`)

2. identify  
   Detect language of input text or file.  
   ```bash
   # uses default model `models/test_model.lid` and default file `input_data/input.txt`
   python cli.py identify [-m <model.lid>] [-t "<text>"] [-f <file.txt>] [--method rank|vector|combined] [-v]
   ```
   - `-m, --model`: path to trained model file (defaults to `models/test_model.lid`)
   - `-t, --text`: text to identify
   - `-f, --file`: path to a `.txt` file (defaults to `input_data/input.txt`)
   - `--method`: comparison method (rank, vector, combined) (default: `vector`)
   - `-v, --verbose`: show detailed scores

3. info  
   Show metadata and configuration of a trained model.  
   ```bash
   # uses default model `models/test_model.lid` if -m is omitted
   python cli.py info [-m <model.lid>]
   ```

## Default Input File
Place a default `.txt` file at `input_data/input.txt` to be used by `identify` when no `-t` or `-f` is specified.

## Examples

```bash
# Train a model (defaults to `dataset/` and outputs `models/test_model.lid`)
python cli.py train

# Or override the output model name:
python cli.py train -o my_model

# Identify using default file (`input_data/input.txt`) and model (`models/test_model.lid`)
python cli.py identify

# Identify with direct text and verbose output:
python cli.py identify -t "This is a test sentence."

# Show model information of default model (`models/test_model.lid`) :
python cli.py info
``` 

## Further Details

For a deep dive into preprocessing, dataset source, algorithmic reference, supported languages,
distance metrics, and assumptions, see [DETAILS.md](DETAILS.md).

## Project Structure

```
.
├── cli.py             # CLI entrypoint
├── config.py          # Configuration defaults
├── core/              # Core library modules
├── dataset/           # Default training samples (subdirectories per language)
├── input_data/        # Default input file for identification (input.txt)
├── models/            # Trained models (*.lid)
└── README.md
```