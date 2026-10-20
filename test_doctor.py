"""doctor 自检的冒烟测试。"""

from personatrace import doctor


def test_doctor_sqlite(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PERSONA_DB", str(tmp_path / "m.db"))
    rc = doctor.main([])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "重启后读回" in out and "记忆链路正常" in out
    assert (tmp_path / "m.db").exists()


def test_doctor_memory_mode(monkeypatch, capsys):
    monkeypatch.setenv("PERSONA_DB", "off")
    rc = doctor.main([])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "进程内后端" in out
