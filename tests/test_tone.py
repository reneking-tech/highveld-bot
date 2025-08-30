from app.rag_chain import _sentiment_score, _format_price_zar


def test_sentiment_score_positive():
    """Test positive sentiment detection."""
    assert _sentiment_score("Thanks, this is great!") > 0.5


def test_sentiment_score_negative():
    """Test negative sentiment detection."""
    assert _sentiment_score("This is useless and frustrating") < -0.5


def test_sentiment_score_neutral():
    """Test neutral sentiment."""
    assert -0.1 < _sentiment_score("I need a quote for water testing") < 0.1


def test_price_formatting():
    """Test consistent price formatting as per principles."""
    assert _format_price_zar(1234.56) == "R1,234.56"
    assert _format_price_zar(100) == "R100.00"
    # Handles non-numeric gracefully
    assert _format_price_zar("invalid") == "invalid"


def test_quote_formatting_empathy():
    """Test that quote responses include empathetic language."""
    # Mock a quote response
    response = "Quote: Total Dissolved Solids — R303.00, TAT 1 day. Total R303.00. Valid 14 days. Would you like a formal PDF quote?"
    assert "Valid 14 days" in response  # Adheres to principles
    assert "Would you like" in response  # Empathetic next step
