from fastapi.testclient import TestClient

from modes.interface import InterfaceStore, conversations, search_excerpt
from modes.server import create_app
from modes.service import ModeService
from test_service import ControlBackend


def test_excerpt_unicode_offsets_and_plain_text():
    value = search_excerpt('前文 😀 STRAẞE 后文 <img src=x onerror=alert(1)>', 'strasse')
    assert value['text'][value['start']:value['end']] == 'STRAẞE'
    assert '<img' in value['text']  # plain text; frontend escapes before marking
    assert search_excerpt('没有这个词', 'no-match') is None


def test_search_returns_exact_message_locator_and_preserves_state(tmp_path):
    service = ModeService(tmp_path, ControlBackend())
    try:
        sid = service.create()["session_id"]
        state = service.turn(sid, '查找这一条：鼻梁红斑，皮肤完整')
        result = conversations(service, q='鼻梁红斑')['items'][0]
        match = result['search_match']
        assert match['text'][match['start']:match['end']] == '鼻梁红斑'
        message = next(row for row in state['messages'] if row['role'] == 'user')
        assert (match['message_at'], match['kind'], match['role']) == (message['at'], message['kind'], 'user')
        assert service.get(sid) == state
        assert 'search_match' not in conversations(service)['items'][0]
        InterfaceStore(tmp_path).update(sid, title='仅标题命中的名称')
        match = conversations(service, q='仅标题命中')['items'][0]['search_match']
        assert match['role'] == 'title' and not match['message_at']
    finally:
        service.close()


def test_search_excerpt_respects_teaching_owner_and_liveness_contract(tmp_path):
    service = ModeService(tmp_path, ControlBackend())
    try:
        sid = service.create('teaching', learning_owner_id='owner-a')['session_id']
        service.turn(sid, 'Private searchable teaching example')
        assert conversations(service, mode='teaching', q='Private', owner_id='owner-b')['items'] == []
        assert conversations(service, mode='teaching', q='Private', owner_id='owner-a')['items'][0]['search_match']
        with TestClient(create_app(service, with_ui=False)) as client:
            assert client.get('/api/health').json()['status'] == 'ok'
            readiness = client.get('/api/readiness').json()
            assert readiness['model'] in {'unknown', 'checking'}
    finally:
        service.close()
