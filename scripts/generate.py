import sys
from pathlib import Path
from aetheris.cli.main import main

if __name__ == "__main__":
    # Simulate arguments if needed, but since we are replacing the script, we can just rely on argparse to parse sys.argv
    # The original script parsed arguments like --prompt, etc.
    # The new CLI expects a subcommand, e.g., 'generate'
    
    # Check if 'generate' is already in argv, if not prepend it
    if len(sys.argv) > 1 and sys.argv[1] != 'generate':
        sys.argv.insert(1, 'generate')
    elif len(sys.argv) == 1:
        sys.argv.append('generate')
        
    sys.exit(main())
