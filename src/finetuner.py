"""Compatibility entry point for the phase 4 SFT CLI.

Use ``python -m training.train_sft`` for new workflows.
"""

from training.train_sft import main


if __name__ == "__main__":
    main()
