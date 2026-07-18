"""Pure differential-memory engine (tracker 3.1)."""

from tools import mem_diff


def _blank():
    return bytearray(0x10000)


def test_classify_region():
    assert mem_diff.classify_region(0x00FE) == "zero_page"
    assert mem_diff.classify_region(0x0400) == "screen_ram"
    assert mem_diff.classify_region(0xD012) == "io_vic"
    assert mem_diff.classify_region(0xD420) == "io_sid"
    assert mem_diff.classify_region(0xD800) == "color_ram"


def test_diff_excludes_io_by_default():
    a, b = _blank(), _blank()
    a[0x00C0] = 5;  b[0x00C0] = 4          # zero-page state change
    a[0xD012] = 10; b[0xD012] = 200        # raster line — volatile I/O
    rows = mem_diff.diff_snapshots(bytes(a), bytes(b))
    addrs = {r["addr"] for r in rows}
    assert 0x00C0 in addrs
    assert 0xD012 not in addrs             # I/O excluded
    r = next(r for r in rows if r["addr"] == 0x00C0)
    assert r["delta"] == -1 and r["region"] == "zero_page"


def test_diff_can_include_io():
    a, b = _blank(), _blank()
    a[0xD012] = 10; b[0xD012] = 11
    rows = mem_diff.diff_snapshots(bytes(a), bytes(b), exclude_io=False)
    assert any(r["addr"] == 0xD012 for r in rows)


def test_diff_region_filter():
    a, b = _blank(), _blank()
    a[0x00C0] = 1; b[0x00C0] = 2           # zero_page
    a[0x0900] = 1; b[0x0900] = 2           # low_ram
    rows = mem_diff.diff_snapshots(
        bytes(a), bytes(b), regions=frozenset({"zero_page"}),
    )
    assert {r["addr"] for r in rows} == {0x00C0}


def test_monotonic_scan_finds_consistent_delta():
    snaps = []
    for i in range(4):
        s = _blank()
        s[0x00C0] = 5 - i                  # lives: 5,4,3,2 (Δ-1 each)
        s[0x0900] = (i * 13) % 256         # noise: inconsistent
        snaps.append(bytes(s))
    hits = mem_diff.monotonic_scan(snaps, delta=-1)
    assert [h["addr"] for h in hits] == [0x00C0]
    assert hits[0]["values"] == [5, 4, 3, 2]


def test_monotonic_scan_needs_two_snapshots():
    assert mem_diff.monotonic_scan([bytes(_blank())], delta=-1) == []


def test_monotonic_scan_ignores_io_region():
    snaps = []
    for i in range(3):
        s = _blank()
        s[0xD012] = 5 - i                  # a decreasing I/O reg — excluded
        snaps.append(bytes(s))
    assert mem_diff.monotonic_scan(snaps, delta=-1) == []


def test_summarize_diff_groups_by_region():
    a, b = _blank(), _blank()
    a[0x00C0] = 1; b[0x00C0] = 2
    text = mem_diff.summarize_diff(mem_diff.diff_snapshots(bytes(a), bytes(b)))
    assert "zero_page" in text and "$00C0" in text
    assert mem_diff.summarize_diff([]).startswith("No differing")
