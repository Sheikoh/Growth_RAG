"""Generic cleaning: structure detection, paragraphs, annotations, speakers.

Source-independent: a source adapter (e.g. sources/gutenberg) removes its own
boilerplate and hands over the body text. Owned books (EPUB, PDF, ...) will have
their own adapter and reuse everything here.

Bump CLEANING_VERSION whenever the rules change: books cleaned with an older
version are re-cleaned automatically by the next `clean` run.
"""

CLEANING_VERSION = "1.2.0"
