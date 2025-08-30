"""State management for conversational context."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class SelectedTest:
    """Represents a selected test with its details."""

    code: str  # Unique identifier for this test
    name: str  # Human-readable name
    price: float  # Price in ZAR (numeric, no currency symbol)
    tat_days: int  # Turnaround time in days


@dataclass
class ChatSession:
    """Manages the state of a chat session for quote flows and escalations."""

    # Session identification
    session_id: str = ""

    # State management
    # idle, selection, selection_confirm, confirmation, pdf, etc.
    state: str = "idle"
    mode: Optional[str] = None  # quote, escalate, etc.

    # Quote flow - selected tests and options
    selected_tests: List[SelectedTest] = field(default_factory=list)
    last_options: List[SelectedTest] = field(default_factory=list)
    pending_quote_id: Optional[str] = None  # Encoded form state
    last_user_free_text: str = ""

    # Escalation flow - consultant request
    escalate_field: Optional[str] = None  # name, contact, time, notes, etc.
    escalate: Dict[str, Any] = field(default_factory=dict)

    # Client details for quote generation
    client_name: str = ""
    client_surname: str = ""
    client_company: str = ""
    client_email: str = ""
    client_extra_info: str = ""
    client_phone: str = ""

    # Sample counts per selected test code
    sample_counts: Dict[str, int] = field(default_factory=dict)
    collect_counts_index: int = 0

    def reset_quote_flow(self) -> None:
        """Reset quote-related state while preserving session identity."""
        self.state = "idle"
        self.selected_tests = []
        self.last_options = []
        self.pending_quote_id = None
        self.last_user_free_text = ""

    def start_escalation(self) -> None:
        """Initialize escalation flow state."""
        self.mode = "escalate"
        self.escalate_field = "name"
        self.escalate = {}
        """Initialize escalation flow state."""
        self.mode = "escalate"
        self.escalate_field = "name"
        self.escalate = {}
