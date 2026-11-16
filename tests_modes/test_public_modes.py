import asyncio
import pytest

from modes.service import ModeService
from test_service import ControlBackend


def test_classic_profiles_exclude_internal_performance():
    from modes import web
    assert set(web.PROFILES.values()) == {"chat", "teaching"}
    assert {profile.name for profile in asyncio.run(web.profiles())} == {"普通问答", "教学模式"}


def test_classic_cannot_restore_internal_run_but_service_can_still_run_it(tmp_path, monkeypatch):
    from modes import web
    service = ModeService(tmp_path, ControlBackend())
    try:
        sid = service.create('performance')['session_id']
        state = service.turn(sid, 'Internal development fixture')
        assert state['performance_result']['answer'] == 'TEST DOUBLE PERFORMANCE'
        monkeypatch.setattr(web, 'get_service', lambda: service)
        with pytest.raises(ValueError, match='内部评测'):
            asyncio.run(web.owned_state(sid))
        assert service.get(sid) == state
    finally:
        service.close()
