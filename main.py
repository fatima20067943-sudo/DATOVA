import json
import logging
import os
import traceback
from dotenv import load_dotenv
import re
from database import SessionLocal
from models import Upload
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
import math
from reportlab.lib.colors import HexColor, white
from reportlab.lib.pagesizes import A4
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
        except Exception:
            raise HTTPException(
                status_code=503,
                detail="AI service is not available. Check GROQ_API_KEY, the model "
                f"name '{MODEL}', and that you have not hit the free rate limit.",
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
    
    db = SessionLocal()
    try:
        upload_record = Upload(
            filename=file.filename,
            status="uploaded",
        )
        db.add(upload_record)
        db.commit()
    except Exception:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Analysis completed, but saving upload information failed.",
        )
    finally:
        db.close()

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
# بناء تقرير الـ PDF: نفس نتائج الشاشة بدون أي تحليل جديد
# (يحتاج: pip install reportlab arabic-reshaper python-bidi + خط عربي داخل fonts/)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
FONT = "DatovaFont"
_font_ready = False

# ---------------------------------------------------------------- الألوان (نفس ثيم الفرونت)
INK = HexColor("#272A43")
PURPLE = HexColor("#6E6AA8")
PURPLE2 = HexColor("#A8A4D4")
LILAC = HexColor("#CBC6EA")
GRAY = HexColor("#777B91")
BODY = HexColor("#5A5D78")
CARD = HexColor("#F5F3FB")
TRACK = HexColor("#E4E0F3")
CHIP = HexColor("#E7E3F5")
WARN = HexColor("#F8DCE6")
PINK = HexColor("#E8B4CB")
PIE = [HexColor(c) for c in ("#6E6AA8", "#E8B4CB", "#8FC3E8", "#A8A4D4", "#B8B9C6", "#E7C6F0")]

M = 44
PAGE_W, PAGE_H = A4
CW = PAGE_W - 2 * M
AR_RE = re.compile(r"[\u0600-\u06FF]")

LABELS = {
    "ar": {
        "title": "تقرير تحليل البيانات", "file": "الملف", "date": "التاريخ",
        "business": "نوع المشروع", "summary": "الخلاصة",
        "rows": "صفوف تم تحليلها", "of": "من",
        "dups": "صفوف مكررة انحذفت", "missing": "صفوف انحذفت لنقص البيانات", "cols": "أعمدة",
        "charts": "المخططات", "insights": "ماذا تقول بياناتك", "recs": "نصائح لك",
        "filled": "تم ملء", "dropped": "تم حذف العمود",
        "mean": "المتوسط", "median": "الوسيط", "mode": "الأكثر تكراراً", "zero": "صفر",
    },
    "en": {
        "title": "Data Analysis Report", "file": "File", "date": "Date",
        "business": "Business type", "summary": "Summary",
        "rows": "Rows analyzed", "of": "of",
        "dups": "Duplicate rows removed", "missing": "Rows removed for missing data", "cols": "columns",
        "charts": "Charts", "insights": "What your data says", "recs": "What to do next",
        "filled": "Filled", "dropped": "Dropped column",
        "mean": "mean", "median": "median", "mode": "most common", "zero": "zero",
    },
}


# ---------------------------------------------------------------- أدوات صغيرة
def register_font() -> None:
    global _font_ready
    if _font_ready:
        return
    dirs = [BASE_DIR / "fonts", BASE_DIR]
    names = ["Amiri-Regular.ttf", "Cairo-Regular.ttf", "NotoNaskhArabic-Regular.ttf"]
    candidates = [d / n for d in dirs for n in names]
    for d in dirs:
        if d.is_dir():
            candidates += sorted(d.glob("*.ttf")) + sorted(d.glob("*.TTF"))
    candidates.append(Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
    for path in candidates:
        if path.exists():
            try:
                pdfmetrics.registerFont(TTFont(FONT, str(path)))
            except Exception:
                continue
            _font_ready = True
            return
    raise RuntimeError("ما لقينا خط عربي. حط Amiri-Regular.ttf داخل مجلد fonts/ جنب main.py")


def _clean(value, limit: int = 2000) -> str:
    return str(value if value is not None else "")[:limit]


def fmt(n) -> str:
    try:
        v = float(n)
    except (TypeError, ValueError):
        return _clean(n, 30)
    return f"{v:,.2f}".rstrip("0").rstrip(".")


def _nums(values) -> list[float]:
    out = []
    for v in values or []:
        try:
            f = float(v)
            out.append(f if math.isfinite(f) else 0.0)
        except (TypeError, ValueError):
            out.append(0.0)
    return out


# ---------------------------------------------------------------- الكاتب
class _Report:
    def __init__(self, fix, lang):
        register_font()
        self.fix = fix
        self.lang = "en" if lang == "en" else "ar"
        self.t = LABELS[self.lang]
        self.rtl = self.lang == "ar"
        self.buf = BytesIO()
        self.c = canvas.Canvas(self.buf, pagesize=A4)
        self.page = 1
        self.y = PAGE_H - M

    # ---- أساسيات
    def w(self, text, size):
        return stringWidth(self.fix(text), FONT, size)

    def right(self, text):
        return self.rtl or bool(AR_RE.search(text))

    def fit(self, text, size, maxw):
        text = _clean(text, 200)
        if self.w(text, size) <= maxw:
            return text
        while text and self.w(text + "...", size) > maxw:
            text = text[:-1]
        return text + "..."

    def put(self, text, x, y, size, color=INK, align="l"):
        self.c.setFillColor(color)
        self.c.setFont(FONT, size)
        s = self.fix(text)
        if align == "r":
            self.c.drawRightString(x, y, s)
        elif align == "c":
            self.c.drawCentredString(x, y, s)
        else:
            self.c.drawString(x, y, s)

    def reg(self, x0, iw, off, width):
        """x اليسار لمنطقة تبدأ على بعد off من الحافة الأمامية (يمين بالعربي، يسار بالإنجليزي)."""
        return x0 + iw - off - width if self.rtl else x0 + off

    def put_lead(self, text, left, width, y, size, color=INK):
        if self.rtl:
            self.put(text, left + width, y, size, color, "r")
        else:
            self.put(text, left, y, size, color, "l")

    def lines(self, text, size, maxw):
        out = []
        for para in _clean(text).split("\n"):
            cur = ""
            for word in para.split():
                trial = f"{cur} {word}".strip()
                if self.w(trial, size) <= maxw:
                    cur = trial
                else:
                    if cur:
                        out.append(cur)
                    cur = word
            out.append(cur)
        return out

    # ---- الصفحات
    def footer(self):
        self.put(f"DATOVA   -   {self.page}", PAGE_W / 2, 24, 8.5, GRAY, "c")

    def new_page(self):
        self.footer()
        self.c.showPage()
        self.page += 1
        self.y = PAGE_H - M

    def need(self, h):
        if self.y - h < M:
            self.new_page()

    # ---- نصوص
    def para(self, text, size=11.5, color=INK, gap=6, indent=0, dot=False):
        text = _clean(text)
        right = self.right(text)
        lh = size * 1.65
        first = True
        for line in self.lines(text, size, CW - indent):
            self.need(lh)
            self.y -= lh
            if right:
                self.put(line, M + CW - indent, self.y, size, color, "r")
            else:
                self.put(line, M + indent, self.y, size, color, "l")
            if dot and first:
                self.c.setFillColor(PURPLE)
                dx = M + CW - 4 if right else M + 4
                self.c.circle(dx, self.y + size * 0.33, 2.4, stroke=0, fill=1)
            first = False
        self.y -= gap

    def section(self, text):
        self.need(54)
        self.y -= 14
        self.para(text, size=15, gap=2)
        self.c.setStrokeColor(LILAC)
        self.c.setLineWidth(1)
        self.c.line(M, self.y, M + CW, self.y)
        self.y -= 10

    # ---- أقسام الصفحة
    def header(self, data):
        t = self.t
        h, pad = 92, 20
        top = self.y
        self.c.setFillColor(INK)
        self.c.roundRect(M, top - h, CW, h, 16, stroke=0, fill=1)
        x0, iw = M + pad, CW - 2 * pad
        half = iw * 0.62
        left = self.reg(x0, iw, 0, half)
        self.put_lead(t["title"], left, half, top - 36, 20, white)
        name = _clean(data.get("original_filename"), 60)
        if name:
            self.put_lead(f"{t['file']}: {name}", left, half, top - 56, 10, LILAC)
        self.put_lead(f"{t['date']}: {datetime.now():%Y-%m-%d %H:%M}", left, half, top - 72, 10, LILAC)
        # العلامة على الجهة الثانية
        trail_w = iw - half - 10
        tl = self.reg(x0, iw, iw - trail_w, trail_w)
        self.put("DATOVA", tl, top - 40, 17, white, "l")
        self.put("Discover What Data Hides.", tl, top - 56, 8, LILAC, "l")
        self.y = top - h - 16

    def business_chip(self, business):
        text = f"{self.t['business']}: {_clean(business, 80) or '-'}"
        wd, h = self.w(text, 11) + 28, 26
        self.need(h + 8)
        left = self.reg(M, CW, 0, wd)
        self.c.setFillColor(CARD)
        self.c.setStrokeColor(LILAC)
        self.c.roundRect(left, self.y - h, wd, h, h / 2, stroke=1, fill=1)
        self.put(text, left + wd / 2, self.y - h + 8, 11, INK, "c")
        self.y -= h + 14

    def summary(self, text):
        text = _clean(text)
        size, pad = 12, 14
        lh = size * 1.7
        lines = self.lines(text, size, CW - 2 * pad)
        h = 2 * pad + size + (len(lines) - 1) * lh + 4
        if h > PAGE_H - 2 * M - 20:
            self.para(text)
            return
        self.need(h + 6)
        top = self.y
        self.c.setFillColor(CARD)
        self.c.setStrokeColor(LILAC)
        self.c.roundRect(M, top - h, CW, h, 12, stroke=1, fill=1)
        right = self.right(text)
        y = top - pad - size
        for ln in lines:
            if right:
                self.put(ln, M + CW - pad, y, size, INK, "r")
            else:
                self.put(ln, M + pad, y, size, INK, "l")
            y -= lh
        self.y = top - h - 12

    def stats(self, rep):
        t = self.t
        gap = 10
        bw, bh = (CW - 3 * gap) / 4, 72
        self.need(bh + 10)
        top = self.y
        items = [
            (t["rows"], rep.get("rows_after"), rep.get("rows_before")),
            (t["dups"], rep.get("duplicates_removed"), None),
            (t["missing"], rep.get("dropped_rows_missing"), None),
            (t["cols"], rep.get("columns_after"), rep.get("columns_before")),
        ]
        for i, (label, val, of) in enumerate(items):
            left = self.reg(M, CW, i * (bw + gap), bw)
            self.c.setFillColor(CARD)
            self.c.setStrokeColor(LILAC)
            self.c.roundRect(left, top - bh, bw, bh, 10, stroke=1, fill=1)
            ty = top - 15
            for ln in self.lines(label, 8.5, bw - 16)[:2]:
                self.put_lead(ln, left + 8, bw - 16, ty, 8.5, GRAY)
                ty -= 11
            self.put_lead(fmt(val or 0), left + 8, bw - 16, top - bh + 26, 18, INK)
            if of is not None:
                self.put_lead(f"{t['of']} {fmt(of)}", left + 8, bw - 16, top - bh + 10, 8.5, GRAY)
        self.y = top - bh - 14

    def chips(self, items):
        if not items:
            return
        size, h, gap = 9, 18, 6
        self.need(h + 8)
        ytop = self.y
        start = M + CW if self.rtl else M
        cx = start
        for text, warn in items:
            text = _clean(text, 80)
            wd = self.w(text, size) + 18
            if (self.rtl and cx - wd < M) or (not self.rtl and cx + wd > M + CW):
                ytop -= h + 6
                if ytop - h < M:
                    self.new_page()
                    ytop = self.y
                cx = start
            left = cx - wd if self.rtl else cx
            self.c.setFillColor(WARN if warn else CHIP)
            self.c.roundRect(left, ytop - h, wd, h, h / 2, stroke=0, fill=1)
            self.put(text, left + wd / 2, ytop - h + 5.5, size, INK, "c")
            cx = left - gap if self.rtl else left + wd + gap
        self.y = ytop - h - 12

    def bullets(self, items):
        for item in items:
            self.para(item, size=11.5, gap=5, indent=16, dot=True)

    # ---- المخططات
    def chart(self, a):
        labels = [_clean(x, 80) for x in (a.get("labels") or [])][:24]
        vals = _nums(a.get("values"))[:24]
        n = min(len(labels), len(vals))
        labels, vals = labels[:n], vals[:n]
        if not n:
            return
        is_pie = a.get("chart") == "pie"
        is_time = n > 1 and all(re.fullmatch(r"\d{4}-\d{2}", l) for l in labels)
        title = _clean(a.get("title") or "-", 150)

        pad = 16
        if is_pie:
            body_h = max(150, n * 20 + 10)
        elif is_time:
            body_h = 165
        else:
            body_h = n * 22
        card_h = 2 * pad + 26 + body_h
        self.need(card_h + 10)
        top = self.y
        self.c.setFillColor(CARD)
        self.c.setStrokeColor(LILAC)
        self.c.roundRect(M, top - card_h, CW, card_h, 14, stroke=1, fill=1)

        x0, iw = M + pad, CW - 2 * pad
        if self.right(title):
            self.put(title, x0 + iw, top - pad - 13, 13, INK, "r")
        else:
            self.put(title, x0, top - pad - 13, 13, INK, "l")
        body_top = top - pad - 26

        if is_pie:
            self._donut(labels, vals, x0, iw, body_top, body_h)
        elif is_time:
            self._vbars(labels, vals, x0, iw, body_top, body_h)
        else:
            self._hbars(labels, vals, x0, iw, body_top)
        self.y = top - card_h - 12

    def _hbars(self, labels, vals, x0, iw, top):
        mx = max(max(abs(v) for v in vals), 1)
        lw, vw = iw * 0.30, 78
        tw = iw - lw - vw - 16
        for i, (lab, v) in enumerate(zip(labels, vals)):
            base = top - i * 22 - 15
            self.put_lead(self.fit(lab, 10, lw), self.reg(x0, iw, 0, lw), lw, base, 10, INK)
            tl = self.reg(x0, iw, lw + 8, tw)
            self.c.setFillColor(TRACK)
            self.c.roundRect(tl, base - 2, tw, 8, 4, stroke=0, fill=1)
            fl = abs(v) / mx * tw
            if fl > 0.5:
                fx = tl + tw - fl if self.rtl else tl
                self.c.setFillColor(PINK if v < 0 else PURPLE2)
                self.c.roundRect(fx, base - 2, fl, 8, min(4, fl / 2), stroke=0, fill=1)
            self.put(fmt(v), self.reg(x0, iw, iw - vw, vw), base, 10, BODY, "l")

    def _vbars(self, labels, vals, x0, iw, top, h):
        n = len(vals)
        gap = 4
        bw = (iw - gap * (n - 1)) / n
        base_y = top - h + 18
        plot_h = h - 34
        mx = max(max(abs(v) for v in vals), 1)
        step = 2 if n > 12 else 1
        for i, (lab, v) in enumerate(zip(labels, vals)):
            bx = x0 + i * (bw + gap)
            bh = max(abs(v) / mx * (plot_h - 12), 2)
            self.c.setFillColor(PINK if v < 0 else PURPLE2)
            self.c.roundRect(bx, base_y, bw, bh, min(3, bw / 2), stroke=0, fill=1)
            if i % step == 0:
                y_, m_ = lab.split("-")
                self.put(f"{m_}/{y_[2:]}", bx + bw / 2, base_y - 12, 7, GRAY, "c")
            txt = fmt(v)
            if n <= 8 and self.w(txt, 7) <= bw + gap:
                self.put(txt, bx + bw / 2, base_y + bh + 3, 7, BODY, "c")

    def _donut(self, labels, vals, x0, iw, top, h):
        total = sum(v for v in vals if v > 0) or 1
        r = min(70, h / 2 - 4)
        cx = x0 + r + 5 if self.rtl else x0 + iw - r - 5
        cy = top - h / 2
        cum = 0.0
        for i, v in enumerate(vals):
            if v <= 0:
                continue
            pct = v / total * 100
            self.c.setFillColor(PIE[i % len(PIE)])
            self.c.setStrokeColor(white)
            self.c.setLineWidth(1)
            if pct >= 99.99:
                self.c.circle(cx, cy, r, stroke=0, fill=1)
            else:
                ext = pct * 3.6
                self.c.wedge(cx - r, cy - r, cx + r, cy + r, 90 - cum * 3.6 - ext, ext, stroke=1, fill=1)
            cum += pct
        self.c.setFillColor(CARD)
        self.c.circle(cx, cy, r * 0.58, stroke=0, fill=1)

        legend_w = iw - 2 * r - 30
        pw = 46
        label_w = legend_w - 16 - pw
        row_top = cy + len(vals) * 10
        for i, (lab, v) in enumerate(zip(labels, vals)):
            base = row_top - i * 20 - 14
            left = self.reg(x0, iw, 0, legend_w)
            dot_x = left + legend_w - 5 if self.rtl else left + 5
            self.c.setFillColor(PIE[i % len(PIE)])
            self.c.circle(dot_x, base + 3.5, 4.5, stroke=0, fill=1)
            self.put_lead(self.fit(lab, 10, label_w), self.reg(x0, iw, 16, label_w), label_w, base, 10, INK)
            self.put(f"{max(v, 0) / total * 100:.1f}%", self.reg(x0, iw, legend_w - pw, pw), base, 10, BODY, "l")

    def finish(self) -> bytes:
        self.footer()
        self.c.save()
        return self.buf.getvalue()


# ---------------------------------------------------------------- الواجهة
def build_report_pdf(data: dict, fix_text) -> bytes:
    """يستلم نفس JSON اللي رجّعه /upload (+ مفتاح lang اختياري) ويرجع bytes مال الـ PDF."""
    r = _Report(fix_text, data.get("lang"))
    t = r.t
    rep = data.get("cleaning_report") if isinstance(data.get("cleaning_report"), dict) else {}

    r.header(data)
    r.business_chip(data.get("business_type"))

    if data.get("summary"):
        r.section(t["summary"])
        r.summary(data["summary"])

    r.stats(rep)

    notes = [(f"{t['filled']}: {_clean(c, 60)} ({t.get(h, h)})", False)
             for c, h in (rep.get("filled_columns") or {}).items()]
    notes += [(f"{t['dropped']}: {_clean(c, 60)}", True) for c in (rep.get("dropped_columns") or [])]
    r.chips(notes[:40])

    analyses = [a for a in (data.get("analyses") or []) if isinstance(a, dict)][:10]
    if analyses:
        r.section(t["charts"])
        for a in analyses:
            r.chart(a)

    insights = [i for i in (data.get("insights") or []) if i][:20]
    if insights:
        r.section(t["insights"])
        r.bullets(insights)

    recs = [i for i in (data.get("recommendations") or []) if i][:20]
    if recs:
        r.section(t["recs"])
        r.bullets(recs)

    return r.finish()

# ---------------------------------------------------------------------------
# endpoint جديد: تحويل نتيجة التحليل إلى PDF
# ---------------------------------------------------------------------------
@app.post("/export-pdf")
def export_pdf(data: dict = Body(...)):
    """يستلم نفس JSON اللي رجّعه /upload ويرجع ملف PDF للتحميل."""
    try:
        pdf_bytes = build_report_pdf(data, fix_text)
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

