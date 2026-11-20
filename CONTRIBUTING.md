# Contributing to mx3sw

Thanks for your interest. Bug reports, questions and pull requests are all welcome.

## Reporting a problem

Open an issue with:

- what you ran (a short script, or the relevant part of your `.mx3` file);
- what you expected and what happened, including the full error message;
- your Python, NumPy and MuMax3 versions.

If the problem is with a particular OVF or `table.txt` file, attaching a small example file helps a lot.

## Setting up

```bash
git clone https://github.com/Dark-atoms/mumax3-spinwave-tools.git
cd mumax3-spinwave-tools
python -m pip install -e ".[test]"
python -m pytest
```

The tests do not need MuMax3 or a GPU; they use small synthetic files and signals.

## Pull requests

- Keep each pull request to one change, and describe what it fixes or adds.
- Add a test for new behaviour or for the bug you fix (see `tests/test_mx3sw.py`).
- Make sure `python -m pytest` passes.
- For physics changes (formulas, sign conventions, units), please give a reference or a short derivation in the description.

Issues labelled **good first issue** are a good place to start.

## License

By contributing you agree that your contributions are licensed under the MIT License of this project.
