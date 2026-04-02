from io import BytesIO

from reportlab.lib.utils import ImageReader, simpleSplit
from reportlab.pdfgen.canvas import Canvas

from src.i18n import get_default_language, translate


PAGE_WIDTH = 612
PAGE_HEIGHT = 792
LEFT_MARGIN = 54
TOP_MARGIN = 54
BOTTOM_MARGIN = 54
CONTENT_WIDTH = PAGE_WIDTH - (LEFT_MARGIN * 2)
LINE_GAP = 5
SPACER_HEIGHT = 10
IMAGE_GAP = 12
MAX_IMAGE_HEIGHT = 260


def _coerce_text(value):
    if value is None:
        return ""
    return str(value)


def _append_text(items, text, *, size=12, bold=False, indent=0):
    items.append(
        {
            "kind": "text",
            "text": text,
            "size": size,
            "bold": bold,
            "indent": indent,
        }
    )


def _append_spacer(items, height=SPACER_HEIGHT):
    items.append({"kind": "spacer", "height": height})


def _append_image(items, image_bytes, *, indent=0):
    items.append(
        {
            "kind": "image",
            "image_bytes": bytes(image_bytes),
            "indent": indent,
        }
    )


def _get_plot_type_label(plot_type, language):
    if not plot_type:
        return ""
    return translate(f"report.plot_type.{plot_type}", language)


def _build_content_items(payload):
    language = payload.get("language") or get_default_language()
    selected_plots = payload.get("selected_plots") or []
    report_sections = payload.get("report_sections") or []

    items = []
    _append_text(items, translate("report.pdf.heading", language), size=18, bold=True)
    _append_spacer(items)

    student_name = _coerce_text(payload.get("student_name") or "")
    group_name = _coerce_text(payload.get("group_name") or "")
    if group_name:
        _append_text(items, f"{translate('report.pdf.group_label', language)} {group_name}")
    if student_name:
        _append_text(items, f"{translate('report.pdf.student_label', language)} {student_name}")

    if student_name or group_name:
        _append_spacer(items)

    _append_text(items, translate("report.pdf.selected_plots", language), size=14, bold=True)
    if selected_plots:
        for plot in selected_plots:
            plot_title = _coerce_text(plot.get("title_text") or plot.get("title", ""))
            plot_type = _coerce_text(plot.get("plot_type", ""))
            plot_type_label = _coerce_text(plot.get("plot_type_label") or _get_plot_type_label(plot_type, language))
            plot_caption = _coerce_text(plot.get("caption_text") or plot.get("caption", ""))
            if plot_title or plot_type_label:
                title_line = plot_title
                if plot_type_label:
                    title_line = f"{title_line} ({plot_type_label})" if title_line else f"({plot_type_label})"
                _append_text(items, f"- {title_line}")
            if plot_caption:
                _append_text(items, plot_caption, indent=18, size=10)
            image_bytes = plot.get("image_bytes")
            if image_bytes:
                _append_image(items, image_bytes)
                _append_spacer(items, IMAGE_GAP)
    else:
        _append_text(items, translate("report.pdf.no_plots_selected", language), indent=18, size=10)

    if report_sections:
        _append_spacer(items)
        _append_text(items, translate("report.pdf.guided_responses", language), size=14, bold=True)
        for section in report_sections:
            section_title = _coerce_text(
                section.get("title_text")
                or translate(section.get("title_msg_id", section.get("step_id", "")), language)
            )
            section_prompt = _coerce_text(
                section.get("prompt_text")
                or translate(section.get("prompt_msg_id", section.get("step_id", "")), language)
            )
            _append_text(items, section_title, bold=True, size=11)
            if section_prompt:
                _append_text(items, section_prompt, indent=18, size=10)

            fields = section.get("fields") or []
            if fields:
                for field in fields:
                    field_label = _coerce_text(
                        field.get("label_text")
                        or translate(field.get("label_msg_id", field.get("field_id", "")), language)
                    )
                    field_answer = _coerce_text(field.get("answer") or "")
                    _append_text(
                        items,
                        f"{field_label}: {field_answer or translate('report.pdf.blank_answer', language)}",
                        indent=18,
                        size=10,
                    )
                continue

            answer = section.get("answer")
            if isinstance(answer, dict):
                for field_id, field_answer in answer.items():
                    answer_text = _coerce_text(field_answer or "") or translate("report.pdf.blank_answer", language)
                    _append_text(items, f"{field_id}: {answer_text}", indent=18, size=10)
            else:
                answer_text = _coerce_text(answer or "") or translate("report.pdf.blank_answer", language)
                _append_text(items, answer_text, indent=18, size=10)

    return items


def _ensure_space(canvas, y, needed_height):
    if y - needed_height < BOTTOM_MARGIN:
        canvas.showPage()
        return PAGE_HEIGHT - TOP_MARGIN
    return y


def _draw_wrapped_text(canvas, y, text, *, size=12, bold=False, indent=0):
    normalized_text = _coerce_text(text)
    font_name = "Helvetica-Bold" if bold else "Helvetica"
    lines = simpleSplit(normalized_text, font_name, size, CONTENT_WIDTH - indent) or [""]

    for line in lines:
        y = _ensure_space(canvas, y, size + LINE_GAP)
        canvas.setFont(font_name, size)
        if line:
            canvas.drawString(LEFT_MARGIN + indent, y, line)
        y -= size + LINE_GAP

    return y


def _draw_image(canvas, y, image_bytes, *, indent=0):
    image_reader = ImageReader(BytesIO(image_bytes))
    image_width, image_height = image_reader.getSize()
    if not image_width or not image_height:
        return y

    scale = min(
        CONTENT_WIDTH / float(image_width),
        MAX_IMAGE_HEIGHT / float(image_height),
        1.0,
    )
    draw_width = image_width * scale
    draw_height = image_height * scale

    y = _ensure_space(canvas, y, draw_height + IMAGE_GAP)
    canvas.drawImage(
        image_reader,
        LEFT_MARGIN + indent,
        y - draw_height,
        width=draw_width,
        height=draw_height,
        preserveAspectRatio=True,
        mask="auto",
    )
    return y - draw_height - IMAGE_GAP


def render_report_pdf(payload):
    items = _build_content_items(payload)
    buffer = BytesIO()
    canvas = Canvas(buffer, pagesize=(PAGE_WIDTH, PAGE_HEIGHT), pageCompression=0)
    y = PAGE_HEIGHT - TOP_MARGIN

    for item in items:
        kind = item["kind"]
        if kind == "text":
            text = item["text"]
            y = _draw_wrapped_text(
                canvas,
                y,
                text,
                size=item["size"],
                bold=item["bold"],
                indent=item["indent"],
            )
        elif kind == "spacer":
            y = _ensure_space(canvas, y, item["height"])
            y -= item["height"]
        elif kind == "image":
            y = _draw_image(canvas, y, item["image_bytes"], indent=item["indent"])

    canvas.save()
    return buffer.getvalue()


def render_report_pdf_safe(payload):
    try:
        return render_report_pdf(payload), None
    except Exception as exc:  # pragma: no cover - exercised through tests
        return None, str(exc)
