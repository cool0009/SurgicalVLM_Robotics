import pytest
from web.video_processor import VideoAnalyzer, _parse_json_result, _normalise_phase, _extract_tool_list, _normalise_tools

def test_parse_json_result_clean():
    """Test JSON parsing with clean JSON input."""
    result = _parse_json_result({'phase': 'Dissection', 'tools': ['Grasper'], 'narrative': 'Cutting'})
    assert result == {'phase': 'Dissection', 'tools': ['Grasper'], 'narrative': 'Cutting'}

def test_parse_json_result_code_fence():
    """Test JSON parsing with code fence input."""
    raw = '```json {"phase": "Preparation", "tools": ["Scalpel"], "narrative": "Making incision"}```'
    result = _parse_json_result(raw)
    assert result['phase'] == 'Preparation'
    assert 'Scalpel' in result['tools']

def test_normalise_phase():
    """Test phase normalization."""
    assert _normalise_phase('Dissection') == 'Dissection'
    assert _normalise_phase('CalotTriangleDissection') == 'CalotTriangle'
    assert _normalise_phase('Unknown') == 'Dissection'

def test_extract_tool_list_str():
    """Test tool list with string entries."""
    result = _extract_tool_list(['Grasper', 'Bipolar'])
    assert result == [('Grasper', 0.9), ('Bipolar', 0.9)]

def test_extract_tool_list_dict():
    """Test tool list with dict entries."""
    result = _extract_tool_list([{'name': 'Hook', 'confidence': 0.8}])
    assert result == [('Hook', 0.8)]

def test_normalise_tools():
    """Test tool normalization."""
    assert _normalise_tools(['Grasper', 'Bipolar']) == ['Grasper', 'Bipolar']
    assert _normalise_tools([{'name': 'Hook', 'confidence': 0.8}]) == ['Hook']