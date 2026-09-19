"""Recover planner SQL column drift against the real Code KB schema."""

import json

import pytest

from code_kb.asm_parser import parse_path
from code_kb.layer0 import build_from_parsed_asm
from code_kb.store import get_code_store
from graph.code_kb_node import code_kb_node


@pytest.fixture()
def code_store(tmp_path):
    asm = tmp_path / "calls.asm"
    asm.write_text(
        "C000  20 00 40  JSR $4000\n"
        "C003  20 FF 47  JSR $47FF\n"
        "C006  20 00 48  JSR $4800\n"
        "C009  60        RTS\n"
        "4000  60        RTS\n"
        "47FF  60        RTS\n"
        "4800  60        RTS\n"
    )
    store = get_code_store(str(tmp_path / "code_kb"))
    build_from_parsed_asm(parse_path(asm), store)
    return store


def run_sql(store, sql):
    result = code_kb_node({
        "code_kb_handle": str(store.root),
        "plan": [{
            "id": "i1_s7", "tool": "code_kb",
            "args": {"mode": "sql", "sql": sql},
        }],
        "current_step_id": "i1_s7",
    })
    return result["tool_results"][0]


def test_recovers_reported_target_addr_query(code_store):
    result = run_sql(
        code_store,
        "SELECT * FROM xrefs WHERE target_addr BETWEEN 16384 AND 18431 "
        "ORDER BY target_addr LIMIT 200",
    )

    assert result["ok"], result["data"]
    assert result["step_id"] == "i1_s7"
    assert result["sql_rewrites_applied"]
    assert json.loads(result["data"]) == code_store.query(
        "SELECT * FROM code_xrefs WHERE dst_addr BETWEEN 16384 AND 18431 "
        "ORDER BY dst_addr LIMIT 200",
    )
    assert [row["dst_addr"] for row in json.loads(result["data"])] == [
        0x4000, 0x47FF,
    ]


def test_recovers_qualified_source_and_target_without_changing_literals(code_store):
    result = run_sql(
        code_store,
        "SELECT x.SOURCE_ADDR, x.TARGET_ADDR, "
        "'source_addr target_addr it''s unchanged' AS note "
        "FROM code_xrefs AS x WHERE x.TARGET_ADDR = 16384",
    )

    assert result["ok"], result["data"]
    assert json.loads(result["data"]) == [{
        "src_addr": 0xC000, "dst_addr": 0x4000,
        "note": "source_addr target_addr it's unchanged",
    }]


def test_valid_sql_aliases_are_not_rewritten(code_store):
    sql = (
        "SELECT src_addr AS source_addr, dst_addr AS target_addr "
        "FROM code_xrefs WHERE dst_addr = 16384 ORDER BY target_addr"
    )
    result = run_sql(code_store, sql)

    assert result["ok"], result["data"]
    assert result["sql"] == sql
    assert "sql_rewrites_applied" not in result
    assert json.loads(result["data"]) == [{
        "source_addr": 0xC000, "target_addr": 0x4000,
    }]
