# fastmove

A CRDT for moving ranges of text, across paragraphs too, with no coordination.
Split, merge, and paragraph moves are special cases. One file, no dependencies.

Every replica applies the same operations in timestamp order, so they converge
whatever order operations arrive in. A late operation undoes the ones after it,
goes in, and they are redone. A move takes its range plus anything typed or
moved into it concurrently. When two moves overlap, the later one wins. A merge
records which paragraph it joins, so a concurrent move can't lose it.

## Use

```python
from fastmove import genesis, Replica

g = genesis("¶abc¶de")
a, b = Replica(1, g), Replica(2, g)
a.move(0, 3, 7)        # drag paragraph 1 below paragraph 2
b.merge(4)             # meanwhile, merge "de" into "abc"
a.receive(b.ops); b.receive(a.ops)
print(a.doc().paragraphs())   # ['abcde'] on both replicas
```

Positions index the visible text. Editors can pass elements instead with
`insert_after`, `erase`, `split_after`, `merge_para` and `move_range`.
## Status

Research code. Tested exhaustively on small documents and with random
histories, not proven correct.

## License

MIT
