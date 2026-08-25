import sys

from .run_suite import main

if __name__ == "__main__":
    sys.argv.insert(1, "whitening")
    main()
