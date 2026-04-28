from adaroute.modules.router import parse_difficulty


def test_router_parse_simple():
    assert parse_difficulty("简单")[0] == "简单"


def test_router_parse_medium():
    assert parse_difficulty("这个任务是中等")[0] == "中等"


def test_router_parse_hard():
    assert parse_difficulty("困难")[0] == "困难"


def test_router_parse_default():
    difficulty, error = parse_difficulty("unknown", "中等")
    assert difficulty == "中等"
    assert error == "ROUTER_PARSE_ERROR"
