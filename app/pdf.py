from __future__ import annotations

from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
import os


BRAND_PRIMARY = colors.HexColor("#0f1e2e")  # deep navy
BRAND_ACCENT = colors.HexColor("#19a3a3")   # teal


def _fmt_price(v: float | int | str) -> str:
    try:
        x = float(v)  # type: ignore[arg-type]
        return f"R{x:,.2f}"
    except Exception:
        return str(v)


def render_quote_pdf(quote: Dict[str, Any], client: Dict[str, Any] | None = None) -> bytes:
    """Render a simple brand-aligned PDF for a quote.

    Expected `quote` shape:
      {
        "tests": [{"test_name": str, "price_ZAR": number, "turnaround_days": number}, ...],
        "total_price_ZAR": number,
        "notes": str,
        "next_step": str,
      }

    `client` can include: name, company, email, phone, reference, billing_address, vat_no.
    """
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18*mm,
                            rightMargin=18*mm, topMargin=16*mm, bottomMargin=16*mm)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        name="Title",
        parent=styles["Title"],
        textColor=BRAND_PRIMARY,
        fontSize=18,
        leading=22,
    )
    h_style = ParagraphStyle(
        name="Heading", parent=styles["Heading2"], textColor=BRAND_PRIMARY)
    body = styles["BodyText"]

    elems: List[Any] = []

    # Header with optional logo
    logo_path = os.path.join(os.path.dirname(
        os.path.dirname(__file__)), "logo.png")
    if os.path.isfile(logo_path):
        try:
            img = Image(logo_path, width=26*mm, height=26*mm)
            elems.append(img)
            elems.append(Spacer(1, 4))
        except Exception:
            pass
    elems.append(Paragraph("Highveld Biotech – Quote", title_style))
    elems.append(Spacer(1, 6))
    elems.append(Paragraph(datetime.now().strftime("%d %b %Y, %H:%M"), body))
    elems.append(Spacer(1, 12))

    # Client block
    client = client or {}
    c_lines = []
    if client.get("name"):
        c_lines.append(f"Client: {client['name']}")
    if client.get("company"):
        c_lines.append(f"Company: {client['company']}")
    if client.get("email"):
        c_lines.append(f"Email: {client['email']}")
    if client.get("phone"):
        c_lines.append(f"Phone: {client['phone']}")
    if client.get("reference"):
        c_lines.append(f"Reference: {client['reference']}")
    if client.get("billing_address"):
        c_lines.append(f"Billing Address: {client['billing_address']}")
    if client.get("vat_no"):
        c_lines.append(f"VAT No: {client['vat_no']}")
    if c_lines:
        elems.append(Paragraph("Client Details", h_style))
        elems.append(Spacer(1, 6))
        for line in c_lines:
            elems.append(Paragraph(line, body))
        elems.append(Spacer(1, 12))

    # Items table
    items = quote.get("tests") or []
    data = [["Test", "Price (ZAR)", "TAT (days)"]]
    for t in items:
        name = t.get("test_name") or t.get("name") or ""
        price = _fmt_price(t.get("price_ZAR", ""))
        tat = t.get("turnaround_days", "")
        data.append([name, price, tat])

    tbl = Table(data, colWidths=[100*mm, 35*mm, 25*mm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BRAND_PRIMARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#dbe2ea")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#f7fafc"), colors.white]),
    ]))
    elems.append(tbl)
    elems.append(Spacer(1, 10))

    # Total
    total = quote.get("total_price_ZAR", 0)
    elems.append(Paragraph(f"<b>Total:</b> {_fmt_price(total)}", body))
    elems.append(Spacer(1, 8))

    # Notes
    notes = (quote.get("notes") or "").strip()
    if notes:
        elems.append(Paragraph("Notes", h_style))
        elems.append(Spacer(1, 4))
        elems.append(Paragraph(notes, body))
        elems.append(Spacer(1, 8))

    # Footer / contact
    elems.append(Spacer(1, 12))
    contact = Paragraph(
        f"<font color='{BRAND_ACCENT}'>labsales@highveldbiotech.com</font> · Highveld Biotech SA PTY Ltd.", body
    )
    elems.append(contact)

    doc.build(elems)
    return buf.getvalue()
