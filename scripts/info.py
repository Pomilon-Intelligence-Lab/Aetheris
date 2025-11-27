import sys
from pathlib import Path
from aetheris.cli.main import main

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != 'info':
        sys.argv.insert(1, 'info')
    elif len(sys.argv) == 1:
        sys.argv.append('info')

    sys.exit(main())
