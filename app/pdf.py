from __future__ import annotations

from datetime import datetime
from io import BytesIO
from typing import List, Dict, Optional

# Optional ReportLab imports; provide a safe fallback if unavailable
try:  # pragma: no cover
    from reportlab.lib.pagesizes import A4  # type: ignore
    from reportlab.lib import colors  # type: ignore
    from reportlab.lib.units import mm  # type: ignore
    from reportlab.pdfgen import canvas  # type: ignore

    _HAS_REPORTLAB = True
except Exception:  # pragma: no cover
    A4 = (595.27, 841.89)  # default A4 size in points
    colors = type("_C", (), {"black": 0, "grey": 0})()  # minimal stub
    mm = 2.83465  # approx conversion
    canvas = None
    _HAS_REPORTLAB = False


def generate_quote_pdf(
    client: Dict,
    items: List[Dict],
    message: str,
    notes: Optional[str] = "",
    *,
    quote_ref: Optional[str] = None,
    valid_until: Optional[str] = None,
    total_price_ZAR: Optional[float] = None,
    disclaimers: Optional[
        str
    ] = "Prices exclude VAT unless otherwise stated. Turnaround time subject to laboratory workload. Quote valid for 14 days. All testing complies with SANS 241 standards.",
) -> bytes:
    """
    Render a simple, branded quote that lists requested tests and the sample counts.
    Falls back to a tiny placeholder PDF if ReportLab is not installed.
    """
    if not _HAS_REPORTLAB:
        # Minimal placeholder PDF bytes so callers can still download something in constrained envs
        ts = datetime.utcnow().isoformat()[:19].replace("T", " ")
        content = f"Highveld Biotech Quote\nGenerated: {ts}\nClient: {client.get('name','')}\nItems: {len(items or [])}\nMessage: {message[:80]}\n"
        # Not a full PDF; but sufficient for tests that only import this module
        return "%PDF-1.1\n%\xe2\xe3\xcf\xd3\n1 0 obj<<>>endobj\n".encode("latin1") + content.encode(
            "utf-8"
        )

    # Real PDF path using ReportLab
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    # Header
    c.setFont("Helvetica-Bold", 14)
    c.drawString(20 * mm, height - 20 * mm, "Highveld Biotech — Quote")
    c.setFont("Helvetica", 10)
    c.drawString(20 * mm, height - 26 * mm, message[:1000])
    # Ref + validity
    y = height - 33 * mm
    c.setFont("Helvetica", 9)
    if quote_ref:
        c.drawString(20 * mm, y, f"Reference: {quote_ref}")
    if valid_until:
        c.drawRightString(190 * mm, y, f"Valid until: {valid_until}")
    y -= 5 * mm

    # Client block
    c.setFont("Helvetica-Bold", 10)
    c.drawString(20 * mm, y, "Client")
    y -= 6 * mm
    c.setFont("Helvetica", 10)
    c.drawString(20 * mm, y, f"Name: {client.get('name','')}")
    y -= 6 * mm
    c.drawString(20 * mm, y, f"Email: {client.get('email','')}")
    y -= 6 * mm
    c.drawString(20 * mm, y, f"Company: {client.get('company','')}")
    y -= 6 * mm
    c.drawString(20 * mm, y, f"Phone: {client.get('phone','')}")
    y -= 6 * mm
    if client.get("reference"):
        c.drawString(20 * mm, y, f"Reference: {client.get('reference')}")
        y -= 6 * mm
    if client.get("billing_address"):
        c.drawString(20 * mm, y, f"Billing address: {str(client.get('billing_address'))[:90]}")
        y -= 6 * mm
    if client.get("vat_no"):
        c.drawString(20 * mm, y, f"VAT: {client.get('vat_no')}")
        y -= 6 * mm

    # Table header
    y -= 2 * mm
    c.setFont("Helvetica-Bold", 11)
    c.drawString(20 * mm, y, "Requested tests")
    y -= 8 * mm
    c.setFont("Helvetica-Bold", 10)
    # Columns: Test | Price | TAT | Qty | Line Total
    c.drawString(20 * mm, y, "Test")
    c.drawRightString(130 * mm, y, "Price")
    c.drawRightString(150 * mm, y, "TAT")
    c.drawRightString(170 * mm, y, "Qty")
    c.drawRightString(190 * mm, y, "Line total")
    y -= 4 * mm
    c.setStrokeColor(colors.black)
    c.line(20 * mm, y, 190 * mm, y)
    y -= 6 * mm

    # Items (use price_ZAR if present; else show as "-" and totals omitted)
    c.setFont("Helvetica", 10)
    running = 0.0
    any_priced = False
    for it in items or []:
        if y < 30 * mm:
            c.showPage()
            width, height = A4
            y = height - 20 * mm
            c.setFont("Helvetica", 10)
        name = str(it.get("name", ""))[:64]
        qty = int(it.get("quantity", 1) or 1)
        price = it.get("price_ZAR", None)
        tat = it.get("turnaround_days", None)
        line_total = None
        if price not in (None, ""):
            try:
                pv = float(price)
                any_priced = True
                line_total = pv * qty
                running += line_total
            except Exception:
                pv, line_total = None, None
        c.drawString(20 * mm, y, name)
        c.drawRightString(
            130 * mm, y, f"R{float(price):,.2f}" if isinstance(price, (int, float)) else "-"
        )
        c.drawRightString(
            150 * mm, y, (str(int(tat)) + " d") if isinstance(tat, (int, float)) else "-"
        )
        c.drawRightString(170 * mm, y, str(qty))
        c.drawRightString(
            190 * mm,
            y,
            f"R{line_total:,.2f}" if isinstance(line_total, (int, float, float)) else "-",
        )
        y -= 6 * mm

    # Totals
    y -= 4 * mm
    if any_priced:
        c.setStrokeColor(colors.black)
        c.line(130 * mm, y, 190 * mm, y)
        y -= 6 * mm
        total = total_price_ZAR if isinstance(total_price_ZAR, (int, float)) else running
        c.setFont("Helvetica-Bold", 10)
        c.drawRightString(170 * mm, y, "Total")
        c.drawRightString(190 * mm, y, f"R{float(total):,.2f}")
        y -= 8 * mm

    # Notes
    if notes:
        c.setFont("Helvetica-Bold", 10)
        c.drawString(20 * mm, y, "Notes")
        y -= 6 * mm
        c.setFont("Helvetica", 10)
        for line in (notes or "").splitlines()[:8]:
            c.drawString(20 * mm, y, line[:110])
            y -= 6 * mm
        y -= 4 * mm

    # Footer
    c.setFont("Helvetica", 8)
    c.setFillColor(colors.grey)
    c.drawString(20 * mm, 15 * mm, disclaimers or "")
    c.setFillColor(colors.black)
    c.showPage()
    c.save()
    return buf.getvalue()
