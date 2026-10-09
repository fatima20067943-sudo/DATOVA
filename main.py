import json
import logging
import os
import traceback
from dotenv import load_dotenv
import re
from pathlib import Path
from uuid import uuid4
import matplotlib
matplotlib.use("Agg")  # بدون شاشة، حتى يشتغل على السيرفر
import matplotlib.pyplot as plt
import pandas as pd
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI
from datetime import datetime
from io import BytesIO

from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

# لدعم العربي داخل الرسومات (pip install arabic-reshaper python-bidi)
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    def fix_text(text) -> str:
        return get_display(arabic_reshaper.reshape(str(text)))
except ImportError:
    def fix_text(text) -> str:
        return str(text)
load_dotenv()
logger = logging.getLogger("datova")
app = FastAPI()


@app.get("/")
def home():
    return FileResponse("index.html")
# Groq (طبقة مجانية). المفتاح يُقرأ من المتغير GROQ_API_KEY
# اسم الموديل: تحقّق منه في صفحة الموديلات بحسابك في موقع Groq لأنه يتغير
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
MODEL = "openai/gpt-oss-120b"
client = AsyncOpenAI(
    base_url=os.environ.get("GROQ_BASE_URL",GROQ_BASE_URL),
    api_key=os.environ.get("GROQ_API_KEY"),
    timeout=60,
)
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
CHARTS_DIR = Path("charts")
CHARTS_DIR.mkdir(exist_ok=True)
app.mount("/charts", StaticFiles(directory=CHARTS_DIR), name="charts")
ALLOWED_EXTENSIONS = {".csv", ".xlsx"}
MAX_FILE_SIZE = 20 * 1024 * 1024
MAX_MISSING_RATIO_FILE = 0.6  # إذا كان أكثر من 60% من الملف فارغًا = مخربط
MAX_MISSING_RATIO_COLUMN = 0.5  # عمود أكثر من 50% فارغًا = نحذفه
ALLOWED_AGG = {"sum", "mean", "count", "max", "min"}
ALLOWED_CHARTS = {"bar", "pie"}
MAX_ANALYSES = 5
MESSY_MESSAGE = (
    "الملف مخربط وما قدرنا نفهم البيانات اللي فيه. "
    "تأكد أن الملف فيه عناوين أعمدة واضحة وبيانات مرتبة وارفعه مرة ثانية."
)


# ---------------------------------------------------------------------------
# قراءة الملف + البروفايل (نفس كودك)
# ---------------------------------------------------------------------------
def read_file(file_path: Path, file_type: str) -> pd.DataFrame:
    try:
        if file_type == ".csv":
            df = pd.read_csv(file_path)
        elif file_type == ".xlsx":
            df = pd.read_excel(file_path)
        else:
            raise ValueError("unsupported file type.")
        return df
    except Exception as error:
        raise ValueError(f"could not read the file: {error}")


def profile_data(df: pd.DataFrame) -> dict:
    return {
        "rows": len(df),
        "columns": list(df.columns),
        "data_types": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "sample_values": {
            column: df[column].dropna().head(5).tolist() for column in df.columns
        },
    }


# ---------------------------------------------------------------------------
# دالة عامة لاستدعاء الذكاء الاصطناعي وإرجاع JSON
# ---------------------------------------------------------------------------
async def ask_json(system_prompt: str, payload: dict) -> dict:
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, default=str),
        },
    ]
    # أحيانًا يرجع الموديل JSON غير صالح، لذلك نجرّب حتى 3 مرات
    for _ in range(3):
        try:
            response = await client.chat.completions.create(
                model=MODEL,
                messages=messages,
                response_format={"type": "json_object"},  # يجبره على إرجاع JSON
                temperature=0,
            )
        except Exception as e:
            logger.exception("Groq API request failed")
            raise HTTPException(
            status_code=503,
            detail=f"AI service error: {type(e).__name__}: {str(e)[:300]}"
            )
        text = (response.choices[0].message.content or "").strip()
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    raise HTTPException(status_code=502, detail="AI returned an invalid response.")


# ---------------------------------------------------------------------------
# Agent 1: يفهم نوع البزنس والملف
# ---------------------------------------------------------------------------
UNDERSTAND_PROMPT = """You are a data-understanding agent.
You receive a profile of a tabular file (columns, dtypes, missing values, sample values).
Decide what kind of business the data belongs to and how it should be cleaned.
Return ONLY valid JSON, no extra text, in this exact shape:
{
"is_understandable": true or false,
"reason_if_not": "short reason, empty string if understandable",
"business_type": "short description in Arabic (e.g. متجر ملابس، مطعم، صيدلية)",
"column_types": {"<column name>": "numeric" | "date" | "category" | "text" | "id"},
"fill_strategy": {"<column name>": "mean" | "median" | "mode" | "zero" | "drop_rows" | "drop_column"}
}
Set is_understandable to false if the columns and values are meaningless, random, or unrelated to any real data.
Use only column names that exist in the profile."""


def looks_too_messy(df: pd.DataFrame) -> bool:
    """فحص سريع بدون AI."""
    if df.size == 0 or len(df) < 5:
        return True
    missing_ratio = df.isnull().sum().sum() / df.size
    unnamed_ratio = sum(str(c).startswith("Unnamed") for c in df.columns) / len(
        df.columns
    )
    return missing_ratio > MAX_MISSING_RATIO_FILE or unnamed_ratio > 0.5


# ---------------------------------------------------------------------------
# التنظيف بواسطة pandas
# ---------------------------------------------------------------------------
def clean_data(df: pd.DataFrame, understanding: dict) -> tuple[pd.DataFrame, dict]:
    report = {
        "rows_before": len(df),
        "columns_before": len(df.columns),
        "filled_columns": {},
        "dropped_columns": [],
        "dropped_rows_missing": 0,
        "duplicates_removed": 0,
    }
    # 1) ترتيب أسماء الأعمدة وحذف الصفوف والأعمدة الفارغة تمامًا
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all").dropna(axis=1, how="all")
    # 2) تنظيف النصوص (مسافات زائدة)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(lambda v: v.strip() if isinstance(v, str) else v)
            df[col] = df[col].replace("", pd.NA)
    # 3) تحويل الأنواع حسب ما فهمه الذكاء الاصطناعي
    column_types = understanding.get("column_types", {})
    for col, col_type in column_types.items():
        if col not in df.columns:
            continue
        if col_type == "numeric":
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "", regex=False),
                errors="coerce",
            )
        elif col_type == "date":
            df[col] = pd.to_datetime(df[col], errors="coerce")
    # 4) حذف التكرار
    before = len(df)
    df = df.drop_duplicates()
    report["duplicates_removed"] = before - len(df)
    # 5) معالجة القيم الفارغة
    fill_strategy = understanding.get("fill_strategy", {})
    for col in list(df.columns):
        missing = int(df[col].isnull().sum())
        if missing == 0:
            continue
        ratio = missing / max(len(df), 1)
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        is_date = pd.api.types.is_datetime64_any_dtype(df[col])
        strategy = fill_strategy.get(col) or ("median" if is_numeric else "mode")
        # عمود أغلبه فارغ: نحذفه
        if strategy == "drop_column" or ratio > MAX_MISSING_RATIO_COLUMN:
            df = df.drop(columns=[col])
            report["dropped_columns"].append(col)
            continue
        # التواريخ لا نملؤها؛ نحذف الصفوف الفارغة
        if is_date or strategy == "drop_rows":
            rows_before = len(df)
            df = df.dropna(subset=[col])
            report["dropped_rows_missing"] += rows_before - len(df)
            continue
        if is_numeric and strategy in {"mean", "median", "zero"}:
            fill_value = {
                "mean": df[col].mean(),
                "median": df[col].median(),
                "zero": 0,
            }[strategy]
            df[col] = df[col].fillna(fill_value)
            report["filled_columns"][col] = strategy
        elif strategy == "mode" and not df[col].mode().empty:
            df[col] = df[col].fillna(df[col].mode().iloc[0])
            report["filled_columns"][col] = "mode"
        else:
            # لم نتمكن من ملئها: نحذف الصفوف
            rows_before = len(df)
            df = df.dropna(subset=[col])
            report["dropped_rows_missing"] += rows_before - len(df)
    df = df.reset_index(drop=True)
    report["rows_after"] = len(df)
    report["columns_after"] = len(df.columns)
    return df, report


# ---------------------------------------------------------------------------
# Agent 2: يخطط التحليل (لا ينفذ كودًا، بل يرجع خطة JSON)
# ---------------------------------------------------------------------------
ANALYSIS_PROMPT = """You are a data-analysis planning agent for small businesses.
You receive the business type and a profile of a CLEAN table.
Plan up to 5 useful analyses that a business owner would care about.
Return ONLY valid JSON, no extra text, in this exact shape:
{
"analyses": [
{
"title": "chart title in Arabic",
"kind": "group_agg" | "monthly_trend",
"group_by": "<existing column name>",
"value": "<existing numeric column name, or null to just count rows>",
"agg": "sum" | "mean" | "count" | "max" | "min",
"top_n": 10,
"chart": "bar" | "pie"
}
]
}
Rules:
- "monthly_trend" requires group_by to be a date column and should use chart "bar".
- Use "pie" only for parts of a whole with few categories (max 6), never for negative values.
- Use only column names that exist in the profile."""


def run_analysis(df: pd.DataFrame, spec: dict) -> dict | None:
    """ينفذ خطة التحليل باستخدام pandas؛ ويتجاهل أي شيء غير مسموح."""
    kind = spec.get("kind", "group_agg")
    group_by = spec.get("group_by")
    value = spec.get("value")
    agg = spec.get("agg", "count")
    chart = spec.get("chart") if spec.get("chart") in ALLOWED_CHARTS else "bar"
    try:
        top_n = max(1, min(int(spec.get("top_n", 10)), 15))
    except (TypeError, ValueError):
        top_n = 10
    if group_by not in df.columns or agg not in ALLOWED_AGG:
        return None
    if value and value not in df.columns:
        return None
    if kind == "monthly_trend":
        if not pd.api.types.is_datetime64_any_dtype(df[group_by]):
            return None
        keys = df[group_by].dt.to_period("M").astype(str)
        chart = "bar"
    else:
        keys = df[group_by].astype(str)
    if agg == "count" or not value:
        series = df.groupby(keys).size()
    else:
        if not pd.api.types.is_numeric_dtype(df[value]):
            return None
        series = df.groupby(keys)[value].agg(agg)
    if kind == "monthly_trend":
        series = series.sort_index().tail(24)
    elif chart == "pie":
        top_n = min(top_n, 6)
        series = series.sort_values(ascending=False).head(top_n)
    if series.empty:
        return None
    if chart == "pie" and (series < 0).any():
        chart = "bar"
    return {
        "title": spec.get("title", "تحليل"),
        "chart": chart,
        "labels": [str(i) for i in series.index],
        "values": [round(float(v), 2) for v in series.values],
    }


# ---------------------------------------------------------------------------
# الرسومات: bar و pie فقط
# ---------------------------------------------------------------------------
def make_chart(result: dict) -> str:
    labels = [fix_text(label) for label in result["labels"]]
    values = result["values"]
    fig, ax = plt.subplots(figsize=(8, 5))
    if result["chart"] == "pie":
        ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
    else:
        ax.bar(labels, values)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title(fix_text(result["title"]))
    fig.tight_layout()
    filename = f"{uuid4().hex}.png"
    fig.savefig(CHARTS_DIR / filename, dpi=120)
    plt.close(fig)
    return f"/charts/{filename}"


# ---------------------------------------------------------------------------
# Agent 3: الاستنتاجات والنصائح
# ---------------------------------------------------------------------------
INSIGHTS_PROMPT = """You are a business advisor for small business owners with no data expertise.
You receive the business type and the results of several analyses (labels and values).
Write in simple Arabic. Base every point ONLY on the numbers you were given; do not invent data.
Return ONLY valid JSON, no extra text, in this exact shape:
{
"summary": "2-3 sentences summarizing the overall picture",
"insights": ["what the data shows, each with a concrete number"],
"recommendations": ["practical, actionable advice"]
}"""


# ---------------------------------------------------------------------------
# الـ endpoint
# ---------------------------------------------------------------------------
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="no file was uploaded")
    file_type = Path(file.filename).suffix.lower()
    if file_type not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="only .csv and .xlsx are allowed",
        )
    safe_filename = f"{uuid4().hex}{file_type}"
    file_path = UPLOAD_DIR / safe_filename
    total_size = 0
    try:
        with open(file_path, "wb") as buffer:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total_size += len(chunk)
                if total_size > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="File size is too large, the max file size is 20 MB",
                    )
                buffer.write(chunk)
    except HTTPException:
        file_path.unlink(missing_ok=True)
        raise
    except Exception:
        file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail="Something went wrong while saving the file.",
        )
    try:
        df = read_file(file_path, file_type)
    except ValueError as error:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(error))
    if df.empty:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    # ---- فحص سريع: الملف مخربط؟ (قبل ما نصرف على الذكاء الاصطناعي) ----
    if looks_too_messy(df):
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=MESSY_MESSAGE)
    # ---- Agent 1: فهم نوع النشاط ----
    profile = profile_data(df)
    understanding = await ask_json(UNDERSTAND_PROMPT, {"profile": profile})
    if not understanding.get("is_understandable", False):
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=MESSY_MESSAGE)
    business_type = understanding.get("business_type", "غير معروف")
    # ---- التنظيف ----
    df, cleaning_report = clean_data(df, understanding)
    if df.empty or len(df.columns) < 2:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=MESSY_MESSAGE)
    # ---- Agent 2: خطة التحليل + التنفيذ باستخدام pandas ----
    plan = await ask_json(
        ANALYSIS_PROMPT,
        {"business_type": business_type, "profile": profile_data(df)},
    )
    results = []
    for spec in plan.get("analyses", [])[:MAX_ANALYSES]:
        result = run_analysis(df, spec)
        if result:
            results.append(result)
    if not results:
        raise HTTPException(
            status_code=422,
            detail="ما قدرنا نطلع تحليلًا مفيدًا من هذا الملف.",
        )
    # ---- الرسومات ----
    for result in results:
        result["chart_url"] = make_chart(result)
    # ---- Agent 3: الاستنتاجات والنصائح ----
    insights = await ask_json(
        INSIGHTS_PROMPT,
        {
            "business_type": business_type,
            "analyses": [
                {k: r[k] for k in ("title", "labels", "values")} for r in results
            ],
        },
    )
    return {
        "message": "File analyzed successfully.",
        "original_filename": file.filename,
        "stored_filename": safe_filename,
        "file_size": total_size,
        "business_type": business_type,
        "cleaning_report": cleaning_report,
        "analyses": results,
        "summary": insights.get("summary", ""),
        "insights": insights.get("insights", []),
        "recommendations": insights.get("recommendations", []),
    }


# ---------------------------------------------------------------------------
# تحويل النتائج إلى PDF (يحتاج: pip install reportlab + خط عربي داخل fonts/)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
FONT_NAME = "ReportFont"

# يجرّب الخطوط بالترتيب ويستخدم أول واحد موجود
FONT_CANDIDATES = [
    BASE_DIR / "fonts" / "Amiri-Regular.ttf",
    BASE_DIR / "fonts" / "Cairo-Regular.ttf",
    BASE_DIR / "fonts" / "NotoNaskhArabic-Regular.ttf",
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),  # احتياطي على لينكس
]

MARGIN = 50
PDF_MAX_ANALYSES = 10
MAX_ITEMS = 20
MAX_TEXT_LEN = 2000

_font_ready = False


def _register_font() -> None:
    """يسجّل الخط مرة وحدة فقط."""
    global _font_ready
    if _font_ready:
        return
    candidates = list(FONT_CANDIDATES)
    # أي ملف .ttf داخل مجلد fonts/ (حتى لو اسمه مختلف عن اللي فوق)
    fonts_dir = BASE_DIR / "fonts"
    if fonts_dir.is_dir():
        candidates[3:3] = sorted(fonts_dir.glob("*.ttf")) + sorted(fonts_dir.glob("*.TTF"))
    # وإذا ما في مجلد fonts: أي ملف .ttf مباشرة جنب main.py
    candidates[3:3] = sorted(BASE_DIR.glob("*.ttf")) + sorted(BASE_DIR.glob("*.TTF"))
    for path in candidates:
        if path.exists():
            try:
                pdfmetrics.registerFont(TTFont(FONT_NAME, str(path)))
            except Exception as error:
                logger.warning("font %s could not be loaded: %s", path, error)
                continue
            _font_ready = True
            logger.info("PDF font loaded: %s", path)
            return
    raise RuntimeError(
        "ما لقينا خط عربي. نزّل Amiri-Regular.ttf وحطه داخل مجلد fonts/ "
        f"(المسار المتوقع: {fonts_dir})"
    )


def _clean_text(value, limit: int = MAX_TEXT_LEN) -> str:
    """يحوّل أي قيمة لنص، ويقصّه حتى ما ينفجر الـ PDF من نص ضخم."""
    return str(value if value is not None else "")[:limit]


def _safe_chart_path(chart_url, charts_dir: Path) -> Path | None:
    """
    نأخذ اسم الملف فقط ونتأكد أنه بالشكل اللي نولّده (32 حرف hex + .png)،
    حتى ما أحد يرسل مسار مثل ../../secret ويقرأ ملفات من السيرفر.
    """
    name = Path(_clean_text(chart_url, 300)).name
    if not re.fullmatch(r"[0-9a-f]{32}\.png", name):
        return None
    path = charts_dir / name
    return path if path.exists() else None


class _PdfWriter:
    """يكتب على الصفحة من فوق لتحت، وينتقل لصفحة جديدة لما تخلص المساحة."""

    def __init__(self, fix_text):
        _register_font()
        self.fix = fix_text
        self.buffer = BytesIO()
        self.pdf = canvas.Canvas(self.buffer, pagesize=A4)
        self.width, self.height = A4
        self.page_number = 1
        self.y = self.height - MARGIN

    @property
    def right_edge(self) -> float:
        return self.width - MARGIN

    @property
    def content_width(self) -> float:
        return self.width - 2 * MARGIN

    def _new_page(self) -> None:
        self._draw_page_number()
        self.pdf.showPage()
        self.page_number += 1
        self.y = self.height - MARGIN

    def _draw_page_number(self) -> None:
        self.pdf.setFont(FONT_NAME, 9)
        self.pdf.drawCentredString(self.width / 2, 25, str(self.page_number))

    def _ensure_space(self, needed: float) -> None:
        if self.y - needed < MARGIN:
            self._new_page()

    def _wrap(self, text: str, size: int, max_width: float) -> list[str]:
        """
        نقسم النص لأسطر بأنفسنا (بدل Paragraph الجاهز) لأن الـ bidi
        لازم يتطبق على كل سطر لحاله، وإلا يطلع ترتيب الأسطر العربية مقلوب.
        """
        lines: list[str] = []
        for paragraph in text.split("\n"):
            current = ""
            for word in paragraph.split():
                trial = f"{current} {word}".strip()
                if stringWidth(self.fix(trial), FONT_NAME, size) <= max_width:
                    current = trial
                else:
                    if current:
                        lines.append(current)
                    current = word
            lines.append(current)
        return lines

    def text(self, text, size: int = 12, space_after: float = 6, bullet: bool = False):
        text = _clean_text(text)
        if bullet:
            text = "• " + text
        line_height = size * 1.6
        self.pdf.setFont(FONT_NAME, size)
        for line in self._wrap(text, size, self.content_width):
            self._ensure_space(line_height)
            self.pdf.setFont(FONT_NAME, size)  # نعيد ضبطه بعد أي صفحة جديدة
            self.y -= line_height
            self.pdf.drawRightString(self.right_edge, self.y, self.fix(line))
        self.y -= space_after

    def heading(self, text, size: int = 16):
        self._ensure_space(size * 3)  # حتى ما يبقى العنوان لحاله بآخر الصفحة
        self.y -= 8
        self.text(text, size=size, space_after=8)

    def image(self, path: Path, max_height: float = 300):
        img = ImageReader(str(path))
        img_w, img_h = img.getSize()
        scale = min(self.content_width / img_w, max_height / img_h)
        draw_w, draw_h = img_w * scale, img_h * scale
        self._ensure_space(draw_h + 10)
        self.y -= draw_h
        x = (self.width - draw_w) / 2
        self.pdf.drawImage(img, x, self.y, draw_w, draw_h)
        self.y -= 14

    def finish(self) -> bytes:
        self._draw_page_number()
        self.pdf.save()
        return self.buffer.getvalue()


def build_pdf(data: dict, charts_dir: Path, fix_text) -> bytes:
    """يبني الـ PDF من نتيجة التحليل ويرجعه كـ bytes."""
    w = _PdfWriter(fix_text)

    # ---- العنوان ----
    w.text("تقرير تحليل البيانات", size=24, space_after=10)
    info = [
        f"الملف: {_clean_text(data.get('original_filename'), 200)}",
        f"نوع النشاط: {_clean_text(data.get('business_type'), 200)}",
        f"التاريخ: {datetime.now():%Y-%m-%d %H:%M}",
    ]
    for line in info:
        w.text(line, size=11, space_after=2)

    # ---- الملخص ----
    if data.get("summary"):
        w.heading("الملخص")
        w.text(data["summary"])

    # ---- التحليلات والرسومات ----
    analyses = data.get("analyses") or []
    for analysis in analyses[:PDF_MAX_ANALYSES]:
        if not isinstance(analysis, dict):
            continue
        w.heading(analysis.get("title", "تحليل"), size=14)
        chart_path = _safe_chart_path(analysis.get("chart_url"), charts_dir)
        if chart_path:
            w.image(chart_path)

    # ---- الاستنتاجات ----
    insights = data.get("insights") or []
    if insights:
        w.heading("أهم الاستنتاجات")
        for item in insights[:MAX_ITEMS]:
            w.text(item, bullet=True)

    # ---- النصائح ----
    recommendations = data.get("recommendations") or []
    if recommendations:
        w.heading("التوصيات")
        for item in recommendations[:MAX_ITEMS]:
            w.text(item, bullet=True)

    # ---- ملخص التنظيف ----
    report = data.get("cleaning_report")
    if isinstance(report, dict):
        w.heading("ملخص تنظيف البيانات", size=14)
        w.text(
            f"عدد الصفوف: من {report.get('rows_before', '?')} إلى {report.get('rows_after', '?')}",
            size=11, bullet=True,
        )
        w.text(f"صفوف مكررة انحذفت: {report.get('duplicates_removed', 0)}", size=11, bullet=True)
        w.text(
            f"صفوف انحذفت بسبب قيم فارغة: {report.get('dropped_rows_missing', 0)}",
            size=11, bullet=True,
        )
        dropped = report.get("dropped_columns") or []
        if dropped:
            w.text(
                "أعمدة انحذفت لأن أغلبها فارغ: " + "، ".join(_clean_text(c, 60) for c in dropped[:20]),
                size=11, bullet=True,
            )

    return w.finish()


# ---------------------------------------------------------------------------
# endpoint جديد: تحويل نتيجة التحليل إلى PDF
# ---------------------------------------------------------------------------
@app.post("/export-pdf")
def export_pdf(data: dict = Body(...)):
    """يستلم نفس JSON اللي رجّعه /upload ويرجع ملف PDF للتحميل."""
    try:
        pdf_bytes = build_pdf(data, CHARTS_DIR, fix_text)
    except RuntimeError as error:  # مثلًا: ما لقينا الخط العربي
        raise HTTPException(status_code=500, detail=str(error))
    except Exception as error:  # أي خطأ ثاني: نطبعه باللوغ ونرجّع سببه للواجهة
        logger.error("PDF export failed:\n%s", traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"PDF export failed: {type(error).__name__}: {error}",
        )
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="report.pdf"'},
    )

