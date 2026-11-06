"""
fastmove.py - the range move of rangemove.py, built to be fast.

It computes the same documents as rangemove.py with every mechanism on
(diff_check.py compares the two, field by field), with different engineering:

  - Incremental. A replica keeps its document and applies each operation once.
    An operation that arrives out of timestamp order undoes the operations after
    it, goes in, and they are redone, as in the replicated tree-move algorithm.
    Each applied operation keeps an undo record for this.
  - The sequence is a doubly linked list. Every element also carries an order
    label, so comparing two positions takes one comparison and placing a range
    needs no scan.
  - Every element keeps the set of elements hanging off it, so a move's closure
    is a search outwards from its range, not repeated passes over the document.
  - An operation's causal history is a version vector, not the set of every
    operation its author had seen.

The API matches rangemove.py (Replica.insert/delete/split/merge/move take
positions in the visible text), and adds cursor-style calls that take elements
(insert_after, erase, split_after, merge_para, move_range), which is what an
editor would use; positions cost a walk over the document.
"""

import heapq
from typing import NamedTuple

HEAD = (0, -1)          # virtual element before everything
END = (0, -2)           # virtual element after everything
BOUND = "¶"        # a paragraph boundary

_TOP = 1 << 62          # label of END; HEAD's is 0
_STEP = 1 << 20         # widest gap left after a newly placed element
_MISSING = object()


class VV:
    """A version vector used as a causal history: (lamport, replica) is in it when
    the author had seen that replica's operations up to that timestamp."""
    __slots__ = ("v",)

    def __init__(self, v=None):
        self.v = dict(v) if v else {}

    def __contains__(self, opid):
        if opid is None or opid[1] < 0:
            return False
        return self.v.get(opid[1], 0) >= opid[0]

    def __repr__(self):
        return "VV(%r)" % (self.v,)


class Op(NamedTuple):
    id: tuple                 # (lamport, replica): unique, and the replay order
    deps: object              # what the author had seen: a VV (or any container of op ids)
    kind: str                 # "ins", "del", "mov" or "mrg"
    elem: tuple = None
    origin: tuple = None
    value: str = None
    members: tuple = ()
    side: str = None
    anchor: tuple = None
    block: bool = False


def concurrent(a, b):
    return a.id != b.id and a.id not in b.deps and b.id not in a.deps


class IllFormed(Exception):
    pass


def _end_of_list(seq, k, op, value, dead):
    """rangemove._end_of on a plain list."""
    deps = op.deps
    j, n = k + 1, len(seq)
    while j < n:
        e = seq[j]
        if value[e] == BOUND and e not in dead and e in deps:
            break
        j += 1
    return j


class Doc:
    """A document, kept up to date one operation at a time."""

    def __init__(self):
        self.nxt = {HEAD: END}
        self.prv = {END: HEAD}
        self.lab = {HEAD: 0, END: _TOP}
        self.dead = set()
        self.value = {}
        self.parent = {}
        self.placer = {}
        self.left_by = {}
        self.last_mover = {}
        self.moved_by = {}
        self.placed = {}
        self.skipped = set()
        self.why_skipped = {}
        self.killed_by = {}
        self.spared = set()
        self.kids = {}            # element -> the elements whose parent it is
        self.byid = {}
        self.applied = []         # ids of the applied operations, ascending
        self.log = []             # their undo records

    # -- reading ------------------------------------------------------------
    @property
    def seq(self):
        out, nxt, x = [], self.nxt, self.nxt[HEAD]
        while x != END:
            out.append(x)
            x = nxt[x]
        return out

    def visible(self):
        dead = self.dead
        return [e for e in self.seq if e not in dead]

    def text(self):
        return "".join(self.value[e] for e in self.visible())

    def paragraphs(self):
        return self.text().split(BOUND)[1:]

    def before(self, a, b):
        """Is element a earlier in the sequence than element b?"""
        return self.lab[a] < self.lab[b]

    # -- the linked list ----------------------------------------------------
    def _unlink(self, e):
        p, s = self.prv[e], self.nxt[e]
        self.nxt[p] = s
        self.prv[s] = p

    def _link_after(self, p, elems):
        """Link elems, in order, right after p."""
        lab, nxt, prv = self.lab, self.nxt, self.prv
        s = nxt[p]
        k = len(elems)
        gap = lab[s] - lab[p]
        if gap <= k:
            self._make_room(p, s, k)
            gap = lab[s] - lab[p]
        # Leave most of the gap after the new elements: text is typed forwards.
        step = gap // (k + 1)
        if step > 64:
            step >>= 6
        if step > _STEP:
            step = _STEP
        label, x = lab[p], p
        for e in elems:
            label += step
            lab[e] = label
            nxt[x] = e
            prv[e] = x
            x = e
        nxt[x] = s
        prv[s] = x

    def _make_room(self, p, s, k):
        """Relabel a stretch around p and s so that k labels fit between them."""
        lab, nxt, prv = self.lab, self.nxt, self.prv
        lo = HEAD if p == HEAD else prv[p]          # the stretch lies strictly between lo and hi
        hi = END if s == END else nxt[s]
        m = (p != HEAD) + (s != END)
        width = 1
        while True:
            sp = (lab[hi] - lab[lo]) // (m + k + 1)
            if sp >= 64 or (lo == HEAD and hi == END):
                break
            for _ in range(width):
                if lo == HEAD:
                    break
                lo = prv[lo]
                m += 1
            for _ in range(width):
                if hi == END:
                    break
                hi = nxt[hi]
                m += 1
            width *= 2
        cur = lab[lo] + (sp * k if p == HEAD else 0)
        x = nxt[lo]
        while x != hi:
            cur += sp
            lab[x] = cur
            if x == p:
                cur += sp * k
            x = nxt[x]

    # -- applying and undoing -----------------------------------------------
    def integrate(self, ops):
        """Apply operations in timestamp order: undo any applied ones that come after the
        first of them, then apply everything in order."""
        ops = sorted(ops, key=lambda o: o.id)
        if not ops:
            return
        first = ops[0].id
        redo = []
        while self.applied and self.applied[-1] > first:
            redo.append(self.byid[self.applied[-1]])
            self.undo_last()
        redo.reverse()
        for op in heapq.merge(redo, ops, key=lambda o: o.id):
            self.apply(op)

    def apply(self, op):
        """Apply one operation that comes after every applied one in timestamp order."""
        self.byid[op.id] = op
        rec = []
        kind = op.kind
        if kind == "ins":
            self._insert(op, rec)
        elif kind == "mrg" and op.members:
            if self._move(op, rec, kill=op.elem) and op.id not in self.spared:
                self._kill(op, rec)
        elif kind in ("del", "mrg"):
            head = self.nxt[HEAD]
            if head != END and head == op.elem and head not in self.dead and self.value[head] == BOUND:
                self._skip(op, "headless", rec)   # it would leave text above every paragraph
            else:
                self._kill(op, rec)
        else:
            self._move(op, rec)
        self.applied.append(op.id)
        self.log.append(rec)

    def undo_last(self):
        """Undo the last applied operation, restoring the document exactly."""
        oid = self.applied.pop()
        rec = self.log.pop()
        value, parent, placer, left_by = self.value, self.parent, self.placer, self.left_by
        last_mover, moved_by, kids = self.last_mover, self.moved_by, self.kids
        for act in reversed(rec):
            kind = act[0]
            if kind == "ins":
                _, x, o = act
                self._unlink(x)
                del self.nxt[x], self.prv[x], self.lab[x], value[x], parent[x], placer[x]
                kids[o].discard(x)
            elif kind == "kill":
                _, x, was_dead, had_killer = act
                if not was_dead:
                    self.dead.discard(x)
                if not had_killer:
                    del self.killed_by[x]
            elif kind == "skip":
                self.skipped.discard(oid)
                del self.why_skipped[oid]
            elif kind == "spare":
                self.spared.discard(oid)
            elif kind == "placed":
                del self.placed[oid]
            elif kind == "left":
                _, e, created = act
                if created:
                    del left_by[e]
                else:
                    left_by[e].discard(oid)
            elif kind == "mem":
                _, e, old_parent, old_placer, old_left = act
                kids[parent[e]].discard(e)
                kids.setdefault(old_parent, set()).add(e)
                parent[e] = old_parent
                placer[e] = old_placer
                if old_left is _MISSING:
                    left_by.pop(e, None)
                else:
                    left_by[e] = old_left
            elif kind == "lm":
                _, e, old = act
                moved_by[e].pop()
                if not moved_by[e]:
                    del moved_by[e]
                if old is _MISSING:
                    del last_mover[e]
                else:
                    last_mover[e] = old
            elif kind == "move":
                _, order, positions = act
                for e in order:
                    self._unlink(e)
                for e, p in positions:
                    self._link_after(p, [e])
        return oid

    def _skip(self, op, why, rec):
        self.skipped.add(op.id)
        self.why_skipped[op.id] = why
        rec.append(("skip",))

    def _insert(self, op, rec):
        x, o = op.elem, op.origin
        self.value[x] = op.value
        self.parent[x] = o
        self.placer[x] = op.id
        self.kids.setdefault(o, set()).add(x)
        self._link_after(o if (o == HEAD or o in self.nxt) else HEAD, [x])
        rec.append(("ins", x, o))

    def _kill(self, op, rec):
        x = op.elem
        rec.append(("kill", x, x in self.dead, x in self.killed_by))
        self.dead.add(x)
        self.killed_by.setdefault(x, op.id)

    def _move(self, op, rec, kill=None):
        """rangemove._move with every mechanism on. Returns whether it was applied."""
        value, dead, byid = self.value, self.dead, self.byid
        parent, placer, left_by, kids = self.parent, self.placer, self.left_by, self.kids
        nxt, prv = self.nxt, self.prv
        deps, oid = op.deps, op.id

        explicit = {e for e in op.members if e in value}
        if op.kind == "mov" and op.members:
            # MERGEWINS: moving a paragraph whose boundary a concurrent merge deleted
            # leaves what that merge moved where the merge put it.
            k = self.killed_by.get(op.members[0])
            if k is not None and byid[k].kind == "mrg" and concurrent(byid[k], op):
                explicit -= set(self.placed.get(k, ()))
        closure = set(explicit)
        if not closure:
            self._skip(op, "empty", rec)
            return False

        # What hangs off the range comes along (CLOSURE, MOVEDIN, DETACH): a search from
        # the range through the elements hanging off it.
        why = {}
        stack = list(closure)
        while stack:
            p = stack.pop()
            ch = kids.get(p)
            if not ch:
                continue
            pex = p in explicit
            for e in ch:
                if e in closure:
                    continue
                src = byid.get(placer[e])
                if src is None:
                    continue
                left = left_by.get(e, ())
                if pex:
                    if not concurrent(src, op):
                        continue
                    if any(m in deps for m in left):
                        continue
                elif left:
                    continue
                closure.add(e)
                why[e] = src.id
                stack.append(e)
        if op.anchor in closure:
            self._skip(op, "cycle", rec)
            return False

        # VIEWORDER: the mover's order, then what came along, placed as it was placed.
        order = [e for e in op.members if e in closure and e not in why]
        if why:
            groups = {}
            for e in sorted(why, key=self.lab.__getitem__):
                groups.setdefault(why[e], []).append(e)
            pending = sorted(groups)
            while pending:
                for key in pending:
                    src = byid[key]
                    anchor = src.origin if src.kind == "ins" else src.anchor
                    if anchor in order:
                        if src.kind != "ins" and src.side == "end":
                            k = _end_of_list(order, order.index(anchor), src, value, dead)
                        elif src.kind != "ins" and src.side == "before" and value.get(anchor) == BOUND \
                                and (anchor in dead or anchor == kill):
                            k = _end_of_list(order, order.index(anchor), src, value, dead)   # MERGEGAP
                        else:
                            k = order.index(anchor) + (0 if src.kind != "ins" and src.side == "before" else 1)
                        order[k:k] = groups[key]
                        pending.remove(key)
                        break
                else:
                    raise IllFormed("move %s: cannot place what came along" % (oid,))

        head = nxt[HEAD]

        def first_rest():
            x = nxt[HEAD]
            while x != END and x in closure:
                x = nxt[x]
            return x

        r0 = first_rest()
        if kill is not None and r0 != END and head == kill and (r0 in dead or value[r0] != BOUND):
            # HEADED: merging the first paragraph, which holds text the merger never saw
            # there. The merged text goes; the boundary stays to head that text.
            held = {kill}
            grew = True
            while grew:
                grew = False
                for e in order:
                    if e not in held and e in why and parent.get(e) in held:
                        held.add(e)
                        grew = True
            order = [e for e in order if e not in held]
            closure -= held
            r0 = first_rest()
            self.spared.add(oid)
            rec.append(("spare",))

        # Where it lands: right after `after`, which is HEAD or an element left in place.
        a, side = op.anchor, op.side
        if a == HEAD:
            after = HEAD
        elif a == END:
            after = prv[END]
            while after != HEAD and after in closure:
                after = prv[after]
        elif a in value and a not in closure:
            if side == "end" or (side == "before" and value[a] == BOUND and (a in dead or a == kill)):
                # the end of a's paragraph: before the next live boundary the author had seen
                after, x = a, nxt[a]
                while x != END:
                    if x not in closure:
                        if value[x] == BOUND and x not in dead and x in deps:
                            break
                        after = x
                    x = nxt[x]
            elif side == "after":
                after = a
            else:
                after = prv[a]
                while after != HEAD and after in closure:
                    after = prv[after]
        else:
            after = HEAD

        # HEADED: never be the step that leaves something above the first boundary.
        gone = kill if (kill is not None and oid not in self.spared) else None
        before_orphan = head != END and (head in dead or value[head] != BOUND)
        nh = order[0] if (after == HEAD and order) else r0
        after_orphan = nh != END and (nh in dead or nh == gone or value[nh] != BOUND)
        if after_orphan and not before_orphan:
            self._skip(op, "headless", rec)
            return False

        moved = sorted(closure, key=self.lab.__getitem__)
        positions = [(e, prv[e]) for e in moved]
        for e in moved:
            self._unlink(e)
        if order:
            self._link_after(after, order)
        rec.append(("move", order, positions))

        last_mover, moved_by = self.last_mover, self.moved_by
        for e in order:
            rec.append(("lm", e, last_mover.get(e, _MISSING)))
            last_mover[e] = oid
            moved_by.setdefault(e, []).append(oid)
        anchor = op.anchor
        for e in op.members:
            if e in closure and e not in why:
                rec.append(("mem", e, parent[e], placer[e], left_by.get(e, _MISSING)))
                kids[parent[e]].discard(e)
                kids.setdefault(anchor, set()).add(e)
                parent[e] = anchor
                placer[e] = oid
                left_by.pop(e, None)
        # DETACH: what hung off the range, was seen by the mover and was not taken stays behind.
        for c in closure:
            ch = kids.get(c)
            if not ch:
                continue
            for e in ch:
                if e not in closure and placer[e] in deps:
                    s = left_by.get(e)
                    if s is None:
                        left_by[e] = {oid}
                        rec.append(("left", e, True))
                    else:
                        s.add(oid)
                        rec.append(("left", e, False))
        self.placed[oid] = order
        rec.append(("placed",))
        return True


def replay(ops, mech=None):
    """Build the document from scratch (same as rangemove.replay with every mechanism on)."""
    d = Doc()
    d.integrate(ops)
    return d


# ---------------------------------------------------------------------------
# A replica: what a user's editor would hold.

class Replica:
    def __init__(self, rid, ops, mech=None):
        self.rid = rid
        self.ops = []
        self.have = set()
        self.vv = {}
        self.clock = 0
        self.mech = mech
        self.d = Doc()
        self.receive(ops)

    def copy(self):
        return Replica(self.rid, self.ops, self.mech)

    def doc(self):
        return self.d

    def receive(self, ops):
        new, have, vv = [], self.have, self.vv
        for o in ops:
            if o.id not in have:
                have.add(o.id)
                self.ops.append(o)
                new.append(o)
                lam, r = o.id
                if lam > self.clock:
                    self.clock = lam
                if lam > vv.get(r, 0):
                    vv[r] = lam
        if new:
            self.d.integrate(new)
        return new

    def _op(self, kind, **kw):
        self.clock += 1
        oid = (self.clock, self.rid)
        if kind == "ins":
            kw["elem"] = oid
        op = Op(oid, VV(self.vv), kind, **kw)
        self.vv[self.rid] = self.clock
        self.have.add(oid)
        self.ops.append(op)
        self.d.apply(op)              # it is the newest operation, so it goes straight in
        return op

    # -- cursor-style edits: positions are elements ---------------------------
    def insert_after(self, elem, value):
        return self._op("ins", origin=elem, value=value)

    def type_after(self, elem, text):
        ops = []
        for ch in text:
            ops.append(self.insert_after(elem, ch))
            elem = ops[-1].elem
        return ops

    def erase(self, elem):
        return self._op("del", elem=elem)

    def split_after(self, elem):
        return self._op("ins", origin=elem, value=BOUND)

    def merge_para(self, bnd):
        """Merge the paragraph that starts at live boundary bnd into the one before it."""
        d = self.d
        value, dead, nxt, prv = d.value, d.dead, d.nxt, d.prv
        prev = prv[bnd]
        while prev != HEAD and not (value[prev] == BOUND and prev not in dead):
            prev = prv[prev]
        assert prev != HEAD, "merge needs a boundary other than the first"
        text, x = [bnd], nxt[bnd]
        while x != END and not (value[x] == BOUND and x not in dead):
            text.append(x)
            x = nxt[x]
        return self._op("mrg", elem=bnd, members=tuple(text), side="end", anchor=prev)

    def move_range(self, first, last, before):
        """Move the visible elements first..last so they sit just before visible element
        `before` (END for the end of the document)."""
        d = self.d
        value, dead, nxt, prv = d.value, d.dead, d.nxt, d.prv
        members, x = [first], first
        while x != last:
            x = nxt[x]
            members.append(x)
        x = nxt[last]
        while x != END and x in dead:
            x = nxt[x]
        if x == END or value[x] == BOUND:
            # The range runs to the end of a paragraph, so it takes the deleted elements there too.
            x = nxt[last]
            while x != END and not (value[x] == BOUND and x not in dead):
                members.append(x)
                x = nxt[x]
        block = any(value[e] == BOUND and e not in dead for e in members)
        if block:
            side, anchor = "before", before
        else:
            side = "after"
            anchor = prv[before] if before != END else prv[END]
            while anchor != HEAD and anchor in dead:
                anchor = prv[anchor]
        return self._op("mov", members=tuple(members), side=side, anchor=anchor, block=block)

    # -- the same calls as rangemove.Replica: positions in the visible text ----
    def insert(self, after, value):
        vis = self.d.visible()
        return [self.insert_after(vis[after] if after >= 0 else HEAD, value)]

    def delete(self, i):
        return [self.erase(self.d.visible()[i])]

    def split(self, after):
        return [self.split_after(self.d.visible()[after])]

    def merge(self, i):
        vis = self.d.visible()
        assert i > 0 and self.d.value[vis[i]] == BOUND, "merge needs a boundary other than the first"
        return [self.merge_para(vis[i])]

    def move(self, i, j, gap):
        vis = self.d.visible()
        return [self.move_range(vis[i], vis[j], END if gap == len(vis) else vis[gap])]


def genesis(text="¶abc¶de"):
    """The starting document, written by replica 0."""
    ops, prev = [], HEAD
    for n, ch in enumerate(text, start=1):
        o = Op((n, 0), VV({0: n - 1} if n > 1 else {}), "ins", elem=(n, 0), origin=prev, value=ch)
        ops.append(o)
        prev = o.elem
    return ops
