"""Command-line interface for language identifier."""

import argparse
import logging
import sys
import os


from core.identifier import LanguageIdentifier
from config import LanguageIdentifierConfig, get_default_config, DEFAULT_CONFIG
from core.exceptions import LanguageIdentifierError

logger = logging.getLogger(__name__)
 
# Default input file for language identification when --text is not provided
DEFAULT_INPUT_FILE = os.path.join(os.path.dirname(__file__), 'input_data', 'input.txt')
# Default dataset directory for training (subdirectories per language)
DEFAULT_DATASET_DIR = os.path.join(os.path.dirname(__file__), 'dataset')
# Default model file for identification and info commands
DEFAULT_MODEL_FILE = os.path.join(os.path.dirname(__file__), 'models', 'test_model.lid')
# Default base name for training output model
DEFAULT_OUTPUT_NAME = os.path.splitext(os.path.basename(DEFAULT_MODEL_FILE))[0]


def setup_parser() -> argparse.ArgumentParser:
    """Set up command-line argument parser."""
    parser = argparse.ArgumentParser(
        description='Language Identifier - Detect the language of text',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new language model')
    train_parser.add_argument('-d', '--dir', default=DEFAULT_DATASET_DIR,
                             help='Directory containing language samples (subdirectories per language)')
    train_parser.add_argument('-o', '--output', default=DEFAULT_OUTPUT_NAME,
                             help='Base name for the output model file (saved as OUTPUT.lid)')
    train_parser.add_argument('--min-ngram', type=int,
                             default=DEFAULT_CONFIG['min_ngram_size'],
                             help='Minimum n-gram size')
    train_parser.add_argument('--max-ngram', type=int,
                             default=DEFAULT_CONFIG['max_ngram_size'],
                             help='Maximum n-gram size')
    train_parser.add_argument('--profile-size', type=int,
                             default=DEFAULT_CONFIG['profile_size'],
                             help='Number of n-grams in language profiles')
    
    # Identify command
    identify_parser = subparsers.add_parser('identify', help='Identify the language of text')
    identify_parser.add_argument('--model', '-m', default=DEFAULT_MODEL_FILE,
                                help='Path to trained model')
    identify_parser.add_argument('--text', '-t',
                                help='Text to identify')
    identify_parser.add_argument('--file', '-f', default=DEFAULT_INPUT_FILE,
                                help='File containing text to identify (default: %(default)s)')
    identify_parser.add_argument('--method',default=DEFAULT_CONFIG['default_method'], choices=['rank', 'vector', 'combined'],
                                help='Comparison method')
    identify_parser.add_argument('--verbose', '-v', action='store_true',
                                help='Show detailed results')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show information about a model')
    info_parser.add_argument('--model', '-m', default=DEFAULT_MODEL_FILE,
                            help='Path to trained model')
    
    return parser


def train_model(args: argparse.Namespace) -> int:
    """Train a new language model."""
    try:
        # Create config
        config = get_default_config()
        config.min_ngram_size = args.min_ngram
        config.max_ngram_size = args.max_ngram
        config.profile_size = args.profile_size
        
        # Create language identifier
        identifier = LanguageIdentifier(config)
        
        # Train on directory
        identifier.train_from_directory(
            directory=args.dir,
            model_name=args.output
        )
        
        print(f"Model trained and saved as '{args.output}'")
        print(f"Supported languages: {', '.join(identifier.get_supported_languages())}")
        return 0
    
    except LanguageIdentifierError as e:
        print(f"Error training model: {e}", file=sys.stderr)
        return 1


def identify_language(args: argparse.Namespace) -> int:
    """Identify the language of text."""
    try:
        # Load model
        identifier = LanguageIdentifier.load_model(args.model)
        
        # Get text from --text or file (falls back to default file when --text not provided)
        if args.text:
            text = args.text
        else:
            file_path = args.file
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            except Exception as e:
                print(f"Error reading file '{file_path}': {e}", file=sys.stderr)
                return 1
        
        # Identify language
        language, confidence, scores = identifier.identify_language(text, args.method)
        
        # Print results
        print()  # empty line for readability
        print(f"Detected language: {language}")
        # Show confidence as percentage
        print(f"Confidence: {confidence:.2%}")
        
        if args.verbose:
            # Pretty-print all language scores in an aligned table
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            # Prepare formatted score strings
            score_strs = [f"{score:.2%}" for _, score in sorted_scores]
            # Determine column widths
            lang_col_width = max(len("Language"), *(len(lang) for lang, _ in sorted_scores))
            score_col_width = max(len("Score"), *(len(s) for s in score_strs))
            # Header and separator
            header = f"{'Language'.ljust(lang_col_width)}  {'Score'.rjust(score_col_width)}"
            separator = f"{'-' * lang_col_width}  {'-' * score_col_width}"
            print("\n" + header)
            print(separator)
            # Rows
            for (lang, _), s in zip(sorted_scores, score_strs):
                print(f"{lang.ljust(lang_col_width)}  {s.rjust(score_col_width)}")
        
        
        return 0
    
    except LanguageIdentifierError as e:
        print(f"Error identifying language: {e}", file=sys.stderr)
        return 1


def show_model_info(args: argparse.Namespace) -> int:
    """Show information about a model."""
    try:
        # Load model
        identifier = LanguageIdentifier.load_model(args.model)
        
        # Get model info
        info = identifier.get_model_info()
        
        # Print info
        print(f"Model: {args.model}")
        print(f"Supported languages: {', '.join(info['languages'])}")
        print(f"Number of languages: {info['num_languages']}")
        
        config = info['config']
        print(f"\nConfiguration:")
        print(f"  Min n-gram size: {config['min_ngram_size']}")
        print(f"  Max n-gram size: {config['max_ngram_size']}")
        print(f"  Profile size: {config['profile_size']}")
        print(f"  Default method: {config['default_method']}")
        
        return 0
    
    except LanguageIdentifierError as e:
        print(f"Error getting model info: {e}", file=sys.stderr)
        return 1


def main() -> int:
    """Main entry point for the CLI."""
    # Suppress INFO-level (and below) log messages for cleaner CLI output
    logging.disable(logging.INFO)
    parser = setup_parser()
    args = parser.parse_args()
    
    if args.command == 'train':
        return train_model(args)
    elif args.command == 'identify':
        return identify_language(args)
    elif args.command == 'info':
        return show_model_info(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())