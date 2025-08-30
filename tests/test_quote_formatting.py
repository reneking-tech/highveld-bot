from app.rag_chain import _render_quote_text


def test_quote_text_formatting():
    """Test quote text formatting for clarity and consistency."""
    obj = {
        "tests": [{"name": "Total Dissolved Solids", "price_ZAR": 303.00, "turnaround_days": 1}],
        "total_price_ZAR": 303.00,
        "clarification_required": False,
    }
    response = _render_quote_text(obj)
    assert "R303.00" in response  # Consistent ZAR formatting
    assert "Valid 14 days" in response  # Validity included
    assert "Would you like a formal PDF quote?" in response  # Next-step orientation


def test_quote_options_formatting():
    """Test formatting when clarification is required."""
    obj = {
        "tests": [],
        "clarification_required": True,
        "options": [
            {"name": "Test 1", "price_ZAR": 100.00, "turnaround_days": 2},
            {"name": "Test 2", "price_ZAR": 200.00, "turnaround_days": 3},
        ],
    }
    response = _render_quote_text(obj)
    assert "1. Test 1 — R100.00 • TAT 2 days" in response
    assert "Please select an option" in response  # Empathetic clarification


def test_multi_test_quote():
    """Test formatting for multiple tests."""
    obj = {
        "tests": [
            {"name": "Test 1", "price_ZAR": 100.00, "turnaround_days": 2},
            {"name": "Test 2", "price_ZAR": 200.00, "turnaround_days": 3},
        ],
        "total_price_ZAR": 300.00,
        "clarification_required": False,
    }
    response = _render_quote_text(obj)
    assert "Total R300.00" in response
    assert "- Test 1 — R100.00, TAT 2 days" in response
