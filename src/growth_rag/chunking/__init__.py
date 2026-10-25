"""Chunking: cleaned book -> retrieval units (chunks).

Source-independent: it reads the cleaning outputs (text + CleanedBook) and the
registry metadata, never raw files. The rule applied depends on each
paragraph's flags set by cleaning, not on the book as a whole:

- prose: paragraphs packed up to a target size, split at sentences if too long;
- poems: one chunk per poem, long poems split at stanza boundaries;
- plays: speeches packed within a scene, never split unless too long alone.

Chunks never cross a section boundary. Bump CHUNKING_VERSION whenever the rules
change: books chunked with an older version are re-chunked automatically.
"""

CHUNKING_VERSION = "1.0.0"
