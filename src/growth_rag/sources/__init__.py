"""Source adapters. Each source (Gutenberg, owned books, ...) knows how to find and
fetch its books, and produces the same BookRecord format in the shared registry,
so later steps never need to know where a book came from."""
