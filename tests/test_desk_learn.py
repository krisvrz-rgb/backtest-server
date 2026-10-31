import json
import random

from app import desk_learn


def _setup(tmp_path, monkeypatch):
    monkeypatch.setattr(desk_learn, "LEARN_FILE", tmp_path / "learn.jsonl")
    monkeypatch.setattr(desk_learn, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(desk_learn, "STATE", {})


def _rows(n, signal, coin="BTC"):
    rnd = random.Random(1)
    for i in range(n):
        f = [rnd.uniform(-1, 1) for _ in range(5)]
        z = signal * (2 * f[2] - f[0])          # orderboek voorspelt, koopdruk 3 min werkt tegen
        y = z + rnd.gauss(0, 1)
        desk_learn.record({"t": i, "coin": coin, "window": 900, "f": f, "s": 0, "p": 0.5, "y": y or 0.01})


def test_learns_real_signal(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    _rows(800, 1.0)
    res = desk_learn.train(900)
    g = res["groups"]["crypto"]
    assert g["use"] and g["ll_learned"] < g["ll_base"]
    assert g["w"][2] > 0.5 and g["w"][0] < -0.2
    assert desk_learn.predict(900, "ETH", {"book": 1, "flow30": -1}) > .8
    assert json.loads((tmp_path / "state.json").read_text())["900"]["n"] == 800


def test_keeps_fixed_model_on_noise(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    _rows(800, 0.0)
    g = desk_learn.train(900)["groups"]["crypto"]
    assert not g["use"]
    assert desk_learn.predict(900, "BTC", {"book": 1}) is None


def test_too_little_data(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    _rows(100, 1.0, coin="xyz:TSLA")
    g = desk_learn.train(900)["groups"]["stock"]
    assert g == {"n": 100, "use": False}
