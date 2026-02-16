"""BOM exporter for CSV/Excel/PDF."""
from __future__ import annotations

import csv
import io
from datetime import datetime

try:
    import openpyxl
    from openpyxl.styles import Font, Alignment, PatternFill
    from openpyxl.utils import get_column_letter
    EXCEL_AVAILABLE = True
except Exception:
    EXCEL_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    PDF_AVAILABLE = True
except Exception:
    PDF_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except Exception:
    PANDAS_AVAILABLE = False

from core.layher_standards import LayherStandards, BillOfMaterials


class BOMExporter:
    def export_to_csv(self, bom: BillOfMaterials, project_name: str = "Unnamed") -> str:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["PROJECT", project_name])
        writer.writerow(["DATE", datetime.now().strftime("%Y-%m-%d %H:%M")])
        writer.writerow([])
        writer.writerow([
            "Article Code", "Description", "Quantity", "Unit Weight (kg)",
            "Total Weight (kg)", "Unit Price (USD)", "Total Price (USD)"
        ])

        for code, count in bom.components.items():
            uw = LayherStandards.ARTICLE_WEIGHTS.get(code, 0)
            up = LayherStandards.ARTICLE_PRICES.get(code, 0)
            writer.writerow([
                code,
                LayherStandards.ARTICLE_NAMES.get(code, "Unknown"),
                count,
                f"{uw:.2f}",
                f"{uw*count:.2f}",
                f"{up:.2f}",
                f"{up*count:.2f}",
            ])

        writer.writerow([])
        writer.writerow(["", "TOTAL", bom.get_total_quantity(), "", f"{bom.get_total_weight():.2f}", "", f"{bom.get_total_cost():.2f}"])
        return output.getvalue()

    def export_to_excel(self, bom: BillOfMaterials, filepath: str, project_name: str = "Unnamed") -> bool:
        if not EXCEL_AVAILABLE:
            return False
        try:
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "BOM"
            ws["A1"] = "BILL OF MATERIALS"
            ws["A1"].font = Font(size=16, bold=True)
            ws["A2"] = f"Project: {project_name}"
            ws["A3"] = f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}"

            headers = ["Article Code", "Description", "Qty", "Unit Weight (kg)", "Total Weight (kg)", "Unit Price (USD)", "Total Price (USD)"]
            row0 = 5
            for c, h in enumerate(headers, start=1):
                cell = ws.cell(row=row0, column=c, value=h)
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
                cell.alignment = Alignment(horizontal="center", vertical="center")

            row = row0 + 1
            for code, count in bom.components.items():
                uw = LayherStandards.ARTICLE_WEIGHTS.get(code, 0)
                up = LayherStandards.ARTICLE_PRICES.get(code, 0)
                ws.cell(row=row, column=1, value=code)
                ws.cell(row=row, column=2, value=LayherStandards.ARTICLE_NAMES.get(code, "Unknown"))
                ws.cell(row=row, column=3, value=count)
                ws.cell(row=row, column=4, value=uw)
                ws.cell(row=row, column=5, value=uw * count)
                ws.cell(row=row, column=6, value=up)
                ws.cell(row=row, column=7, value=up * count)
                row += 1

            row += 1
            ws.cell(row=row, column=1, value="TOTAL").font = Font(bold=True)
            ws.cell(row=row, column=3, value=bom.get_total_quantity())
            ws.cell(row=row, column=5, value=bom.get_total_weight())
            ws.cell(row=row, column=7, value=bom.get_total_cost())

            for col in range(1, 8):
                ws.column_dimensions[get_column_letter(col)].width = 18
            wb.save(filepath)
            return True
        except Exception:
            return False

    def export_to_pdf(self, bom: BillOfMaterials, filepath: str, project_name: str = "Unnamed") -> bool:
        if not PDF_AVAILABLE:
            return False
        try:
            doc = SimpleDocTemplate(filepath, pagesize=A4)
            story = []
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=18, textColor=colors.HexColor("#1f4788"), spaceAfter=12)
            story.append(Paragraph("BILL OF MATERIALS", title_style))
            story.append(Spacer(1, 0.3 * cm))
            story.append(Paragraph(f"<b>Project:</b> {project_name}", styles["Normal"]))
            story.append(Paragraph(f"<b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
            story.append(Spacer(1, 0.5 * cm))

            rows = [["Article", "Description", "Qty", "Weight/unit", "Total Weight", "Price/unit", "Total Price"]]
            for code, count in bom.components.items():
                uw = LayherStandards.ARTICLE_WEIGHTS.get(code, 0)
                up = LayherStandards.ARTICLE_PRICES.get(code, 0)
                rows.append([code, LayherStandards.ARTICLE_NAMES.get(code, "Unknown"), str(count), f"{uw:.1f} kg", f"{uw*count:.1f} kg", f"${up:.2f}", f"${up*count:.2f}"])
            rows.append(["", "TOTAL", str(bom.get_total_quantity()), "", f"{bom.get_total_weight():.1f} kg", "", f"${bom.get_total_cost():.2f}"])

            table = Table(rows, repeatRows=1)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#4472C4")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BACKGROUND', (0, -1), (-1, -1), colors.beige),
                ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            story.append(table)
            doc.build(story)
            return True
        except Exception:
            return False

    def export_to_dataframe(self, bom: BillOfMaterials):
        if not PANDAS_AVAILABLE:
            return None
        rows = []
        for code, count in bom.components.items():
            uw = LayherStandards.ARTICLE_WEIGHTS.get(code, 0)
            up = LayherStandards.ARTICLE_PRICES.get(code, 0)
            rows.append({
                "article_code": code,
                "description": LayherStandards.ARTICLE_NAMES.get(code, "Unknown"),
                "quantity": count,
                "unit_weight_kg": uw,
                "total_weight_kg": uw * count,
                "unit_price_usd": up,
                "total_price_usd": up * count,
            })
        return pd.DataFrame(rows)
