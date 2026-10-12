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
from pydantic import BaseModel
from typing import Literal
from database import SessionLocal
from models import Upload
from datetime import datetime
from io import BytesIO
import unicodedata
import warnings
import asyncio
import time as _time
from datetime import time as dt_time
from difflib import SequenceMatcher
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
# === التنظيف (نسخة أقوى) ===================================================
# كل الأرقام اللي تتحكم بسلوك التنظيف هنا، غيّرها من مكان واحد
UNKNOWN_LABEL = "غير محدد"        # يُملأ فيه النص الناقص
MAX_FILL_RATIO = 0.15             # رقم ناقص أكثر من 15%: ما نملاه (الملء بالوسيط يضخّم المجاميع)
MAX_ROW_LOSS = 0.4                # ما نحذف صفوف إذا الحذف ياكل أكثر من 40% من الملف
TYPE_THRESHOLD = 0.9              # نسبة نجاح التحويل المطلوبة حتى نغيّر نوع العمود
HINT_THRESHOLD = 0.8              # إذا الذكاء الاصطناعي حدد النوع (رقم/تاريخ) نكتفي بنسبة أقل
FUZZY_THRESHOLD = 0.85            # تشابه الكلمات لدمجها (0.85 = متحفظ، أقل = أجرأ)
FUZZY_MIN_LEN = 4                 # الكلمات الأقصر من هذا ما تنقارن تقريبياً (A و B ما تندمج)
MAX_FUZZY_UNIQUE = 1000           # أكثر من هذا العدد من القيم المختلفة: نوحّد الحروف بس بدون مقارنة تقريبية
FUZZY_TIME_BUDGET = 1.5           # أقصى ثواني للمقارنة التقريبية لكل عمود (حماية من البطء)
FUZZY_TOTAL_BUDGET = 4.0          # وأقصى ثواني لكل الملف
MAX_STANDARD_LEN = 40             # أعمدة نصوصها طويلة (ملاحظات) ما نوحّد كلماتها
DATE_DAYFIRST = True              # 05/03/2026 = 5 مارس
DEDUP_MODE = "auto"               # "auto": نحذف المكرر إذا الجدول فيه معرّف/تاريخ/ساعة | "always" | "never"
MISSING_TOKENS = {"", "nan", "null", "n/a", "#n/a", "-", "--", "غير متوفر"}

_SPACES = re.compile(r"\s+")
_ZERO_WIDTH = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")
_TIME_PAT = r"^\d{1,2}:\d{2}(?::\d{2})?\s*(?:[AaPp][Mm]|ص|م)?$"
_DATE_PAT = r"^\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}"
_OFFSET_PAT = r"(?<=\d:\d{2})\s*(?:Z|[+-]\d{2}:?\d{2})$"      # فرق التوقيت (+03:00 / Z) نشيله ونحافظ على الساعة المحلية
_CURRENCY_PAT = r"(?i)(?:iqd|usd|eur|د\.ع|دينار|دولار|[$€£%])"
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹٫٬", "01234567890123456789.,")
_AR_MAP = str.maketrans({"ة": "ه", "ى": "ي", "ی": "ي", "ک": "ك", "ـ": ""})
_PD2 = int(pd.__version__.split(".")[0]) >= 2


def _clean_cell(v):
    """يشيل المسافات الزايدة (بالأول والآخر ووسط الكلمات) والرموز المخفية، ويحوّل 'nan' و'-' وغيرها لقيمة ناقصة."""
    if not isinstance(v, str):
        return v
    s = _SPACES.sub(" ", _ZERO_WIDTH.sub("", v)).strip()
    return pd.NA if s.lower() in MISSING_TOKENS else s


def _clean_columns(cols) -> list[str]:
    """أسماء أعمدة نظيفة، غير فارغة، وغير مكررة."""
    out, seen = [], {}
    for i, c in enumerate(cols, 1):
        name = _SPACES.sub(" ", _ZERO_WIDTH.sub("", str(c))).strip() or f"column_{i}"
        seen[name] = seen.get(name, 0) + 1
        out.append(name if seen[name] == 1 else f"{name}_{seen[name]}")
    return out


def _to_datetime(s: pd.Series) -> pd.Series:
    """2026-01-05 (سنة أول) تنقرأ سنة-شهر-يوم، و05/01/2026 تنقرأ يوم/شهر/سنة (حسب DATE_DAYFIRST)."""
    s = s.astype("string").str.replace(_OFFSET_PAT, "", regex=True)
    year_first = s.str.match(r"^\s*\d{4}[-/.]").fillna(False)

    def parse(part, dayfirst):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                kwargs = {"errors": "coerce", "dayfirst": dayfirst}
                if _PD2:
                    kwargs["format"] = "mixed"
                return pd.to_datetime(part, **kwargs)
            except Exception:
                return pd.Series(pd.NaT, index=part.index, dtype="datetime64[ns]")

    return parse(s.where(year_first), False).fillna(parse(s.where(~year_first), DATE_DAYFIRST))


def _norm_num(s: str):
    """يفهم 1,234.56 و 1.234,56 و 12,5 و (500) و 500-  ويرجع نص رقم صالح أو None."""
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg, s = True, s[1:-1]
    if s.startswith("-"):
        neg, s = True, s[1:]
    elif s.startswith("+"):
        s = s[1:]
    if s.endswith("-"):
        neg, s = True, s[:-1]
    if "," in s and "." in s:                       # الفاصل الأخير هو العشري
        dec = "," if s.rfind(",") > s.rfind(".") else "."
        grp = "." if dec == "," else ","
        s = s.replace(grp, "").replace(dec, ".")
    elif "," in s:
        if re.fullmatch(r"\d{1,3}(?:,\d{3})+", s):  # 1,234 أو 12,500 = آلاف
            s = s.replace(",", "")
        elif re.fullmatch(r"\d+,\d{1,2}", s):       # 12,5 أو 7,25 = عشري
            s = s.replace(",", ".")
        else:
            return None
    elif s.count(".") > 1:
        if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", s):  # 1.234.567 = آلاف
            s = s.replace(".", "")
        else:
            return None
    return ("-" if neg else "") + s if re.fullmatch(r"\d+(?:\.\d+)?", s) else None


def _detect(series: pd.Series, hint: str | None):
    """يرجع (النوع, العمود المحوّل, mask بالقيم اللي فشل تحويلها). الأنواع: numeric/date/time/id/bool/text."""
    if pd.api.types.is_bool_dtype(series):
        return "bool", series, None
    if hint == "id":
        return "id", series, None
    if pd.api.types.is_numeric_dtype(series):
        return "numeric", series, None
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date", series, None
    nn = series.dropna()
    if nn.empty:
        return "text", series, None
    if all(isinstance(v, dt_time) for v in nn):
        return "time", series, None
    txt = series.astype("string")
    present = txt.notna()
    n = int(present.sum())

    # --- ساعة (14:30 / 2:15 PM / 9:00 ص)
    is_time = txt.str.match(_TIME_PAT).fillna(False)
    if is_time.sum() / n >= 0.8:
        fixed = txt.str.replace(r"\s*ص$", " AM", regex=True).str.replace(r"\s*م$", " PM", regex=True)
        parsed = _to_datetime(fixed.where(is_time))
        if parsed.notna().sum() / n >= TYPE_THRESHOLD:
            return "time", parsed.dt.time, present & parsed.isna()

    # --- تاريخ
    date_like = txt.str.match(_DATE_PAT).fillna(False)
    if hint == "date" or date_like.sum() / n >= 0.8:
        parsed = _to_datetime(txt)
        if parsed.notna().sum() / n >= (HINT_THRESHOLD if hint == "date" else TYPE_THRESHOLD):
            return "date", parsed, present & parsed.isna()

    # --- رقم (أرقام عربية، آلاف/عشري، عملات، أقواس سالبة)
    cleaned = (txt.str.translate(_DIGITS)
                  .str.replace(_CURRENCY_PAT, "", regex=True)
                  .str.replace(r"\s", "", regex=True))
    normed = cleaned.map({u: _norm_num(u) for u in cleaned.dropna().unique()})
    ok = normed.notna()
    if ok.sum() / n >= (HINT_THRESHOLD if hint == "numeric" else TYPE_THRESHOLD):
        if cleaned[ok].str.match(r"^0\d+$").mean() > 0.3:   # 0770... أرقام هواتف/معرّفات: تبقى نص
            return "id", series, None
        num = pd.to_numeric(normed.where(ok), errors="coerce").astype("float64")
        if num.notna().all() and (num % 1 == 0).all():
            num = num.astype("int64")
        return "numeric", num, present & ~ok
    return "text", series, None


def _tokens(text: str) -> list[str]:
    s = unicodedata.normalize("NFKD", text)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.findall(r"\w+", s.translate(_AR_MAP).casefold())


def _key(text: str) -> str:
    """مفتاح مقارنة. للنص: بدون حروف كبيرة/صغيرة، تشكيل، همزات، ة/ه، ى/ي، مسافات ورموز.
    للقيم اللي فيها أرقام (أكواد، 1-2): نوحّد الحروف والمسافات فقط، وما نشيل الرموز."""
    strict = _SPACES.sub(" ", text).strip().casefold()
    if any(ch.isdigit() for ch in text):
        return strict
    loose = "".join(_tokens(text))
    return loose if len(loose) >= 3 else strict        # C++ غير C#


def _tokens_close(a: list[str], b: list[str], sm: SequenceMatcher) -> bool:
    """نفس عدد الكلمات، وكل كلمة مختلفة لازم تكون طويلة ومتشابهة. (Category A و Category B ما يندمجون)"""
    if len(a) != len(b):
        return False
    differs = False
    for x, y in zip(a, b):
        if x == y:
            continue
        if len(x) < FUZZY_MIN_LEN or len(y) < FUZZY_MIN_LEN or any(ch.isdigit() for ch in x + y):
            return False
        sm.set_seqs(x, y)
        if sm.real_quick_ratio() < FUZZY_THRESHOLD or sm.quick_ratio() < FUZZY_THRESHOLD or sm.ratio() < FUZZY_THRESHOLD:
            return False
        differs = True
    return differs


def _standardize_column(s: pd.Series, deadline: float) -> tuple[pd.Series, dict]:
    """يوحّد الكلمات المكتوبة بأكثر من طريقة (Fatima / fatma / FATIMA / فاطمة / فاطمه) على أكثر شكل مكرر."""
    vals = s.dropna()
    vals = vals[vals.map(lambda v: isinstance(v, str))]
    if vals.empty or vals.map(len).mean() > MAX_STANDARD_LEN:
        return s, {}
    counts = vals.value_counts()

    # 1) تجميع الأشكال المتطابقة بعد التطبيع
    groups: dict[str, list] = {}
    for original, c in counts.items():
        groups.setdefault(_key(original) or original, []).append((original, int(c)))

    # 2) دمج الكلمات المتقاربة (أخطاء إملائية) بحذر
    order = sorted(groups, key=lambda k: -sum(c for _, c in groups[k]))
    parent: dict[str, str] = {}
    if len(order) <= MAX_FUZZY_UNIQUE:
        sm = SequenceMatcher(autojunk=False)
        toks: dict[str, list[str]] = {}
        buckets: dict[str, list[str]] = {}               # نقارن فقط الكلمات اللي تبدأ بنفس الحرف (أسرع بكثير)
        for k in order:
            if _time.perf_counter() > deadline:
                break
            if len(k) < FUZZY_MIN_LEN or any(ch.isdigit() for ch in k):
                continue
            toks[k] = _tokens(groups[k][0][0])
            target, best = None, 0.0
            for c in buckets.get(k[0], []):
                if abs(len(c) - len(k)) > 2 or not _tokens_close(toks[c], toks[k], sm):
                    continue
                r = SequenceMatcher(None, c, k).ratio()
                if r > best:
                    best, target = r, c
            if target is None:
                buckets.setdefault(k[0], []).append(k)
            else:
                parent[k] = target

    members: dict[str, list] = {}
    for k, items in groups.items():
        members.setdefault(parent.get(k, k), []).extend(items)

    # 3) الشكل النهائي = الأكثر تكراراً (وعند التعادل نفضّل الشكل Capitalized)
    mapping = {}
    for items in members.values():
        if len(items) < 2:
            continue
        best_form = sorted(items, key=lambda x: (-x[1], not x[0].istitle(), x[0]))[0][0]
        for original, _ in items:
            if original != best_form:
                mapping[original] = best_form
    if not mapping:
        return s, {}
    return s.map(lambda v: mapping.get(v, v) if isinstance(v, str) else v), mapping


def clean_data(df: pd.DataFrame, understanding: dict) -> tuple[pd.DataFrame, dict]:
    report = {
        "rows_before": len(df),
        "columns_before": len(df.columns),
        "filled_columns": {},        # عمود: طريقة الملء (mean/median/zero/unknown)
        "filled_cells": {},          # عمود: كم خلية انملت
        "dropped_columns": [],       # أعمدة فاضية تماماً فقط
        "dropped_rows_missing": 0,
        "duplicates_removed": 0,
        "duplicates_kept": 0,        # صفوف متطابقة تركناها لأن ما اكو معرّف/تاريخ يثبت إنها تكرار
        "converted_columns": {},     # عمود: النوع اللي تحوّل إله (numeric/date/time)
        "conversion_losses": {},     # عمود: كم قيمة ما قدرنا نحوّلها وصارت فاضية
        "standardized_columns": {},  # عمود: {unique_before, unique_after, examples}
        "sparse_columns": [],        # أعمدة أغلبها فاضي: انحفظت بدون ملء وبدون حذف صفوف
    }
    df = df.copy()
    understanding = understanding if isinstance(understanding, dict) else {}

    # 1) أسماء الأعمدة (ونربط تعليمات الذكاء الاصطناعي بالأسماء الجديدة)
    old = [str(c) for c in df.columns]
    df.columns = _clean_columns(df.columns)
    rename = dict(zip(old, df.columns))

    def remap(d):
        return {rename.get(str(k), str(k)): v for k, v in d.items()} if isinstance(d, dict) else {}

    types = remap(understanding.get("column_types"))
    strategy = remap(understanding.get("fill_strategy"))

    # 2) تنظيف الخلايا النصية: مسافات زايدة، رموز مخفية، 'nan' و'-' = ناقص
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_bool_dtype(s):
            continue
        if isinstance(s.dtype, pd.CategoricalDtype):
            s = s.astype(object)
        df[col] = s.map(_clean_cell)

    # 3) حذف الصفوف الفاضية تماماً + الأعمدة الفاضية تماماً (فقط)
    df = df.dropna(how="all")
    empty = [c for c in df.columns if df[c].isna().all()]
    if empty:
        df = df.drop(columns=empty)
        report["dropped_columns"].extend(empty)

    # 4) تحويل الأنواع: رقم = رقم، تاريخ = تاريخ، ساعة = ساعة، والنص يبقى نص
    kinds = {}
    protected = pd.Series(False, index=df.index)   # صفوف فيها قيمة فشل تحويلها: ما نعتبرها مكررة
    for col in df.columns:
        was_object = df[col].dtype == object or pd.api.types.is_string_dtype(df[col])
        kind, converted, lost = _detect(df[col], types.get(col))
        kinds[col] = kind
        if kind in {"numeric", "date", "time"} and was_object:
            df[col] = converted
            report["converted_columns"][col] = kind
            if lost is not None and int(lost.sum()) > 0:
                report["conversion_losses"][col] = int(lost.sum())
                protected |= lost.reindex(df.index, fill_value=False).astype(bool)

    # 5) توحيد الكلمات المكتوبة بأكثر من طريقة (قبل حذف التكرار حتى تنكشف الصفوف المكررة)
    total_deadline = _time.perf_counter() + FUZZY_TOTAL_BUDGET
    for col in df.columns:
        if kinds[col] != "text":
            continue
        before = int(df[col].nunique())
        new_col, mapping = _standardize_column(df[col], min(total_deadline, _time.perf_counter() + FUZZY_TIME_BUDGET))
        if mapping:
            df[col] = new_col
            report["standardized_columns"][col] = {
                "unique_before": before,
                "unique_after": int(df[col].nunique()),
                "examples": dict(list(mapping.items())[:8]),
            }

    # 6) المكرر: نحذفه فقط إذا اكو دليل إنه تكرار فعلاً (معرّف/تاريخ/ساعة)، وإلا نتركه ونبلّغ عنه
    try:
        dup = df.duplicated(keep="first") & ~protected
    except TypeError:  # قيم غير قابلة للمقارنة
        dup = pd.Series(False, index=df.index)
    has_identity = any(k in {"id", "date", "time"} for k in kinds.values()) or any(
        kinds[c] == "numeric" and pd.api.types.is_integer_dtype(df[c]) and len(df) > 0
        and df[c].nunique() >= 0.95 * len(df)
        for c in df.columns
    )
    if DEDUP_MODE == "always" or (DEDUP_MODE == "auto" and has_identity):
        report["duplicates_removed"] = int(dup.sum())
        df = df[~dup]
    else:
        report["duplicates_kept"] = int(dup.sum())

    # 7) القيم الفارغة: نملأ إذا نكدر، وإذا ما نكدر نحذف الصف (بحدود تمنع خراب الملف)
    n = max(len(df), 1)
    drop_cols = []
    for col in df.columns:
        miss = int(df[col].isna().sum())
        if miss == 0:
            continue
        ratio, kind, how = miss / n, kinds[col], strategy.get(col)
        if kind == "numeric":
            if how == "drop_rows":
                drop_cols.append(col)
            elif ratio <= MAX_FILL_RATIO:
                how = how if how in {"mean", "median", "zero"} else "median"
                value = {"mean": df[col].mean(), "median": df[col].median(), "zero": 0}[how]
                df[col] = df[col].fillna(0 if pd.isna(value) else value)
                report["filled_columns"][col] = how
                report["filled_cells"][col] = miss
            else:
                report["sparse_columns"].append(col)   # أغلبه فاضي: نبقيه كما هو، والتحليل يتجاهل الفاضي
        elif kind == "text":
            df[col] = df[col].fillna(UNKNOWN_LABEL)
            report["filled_columns"][col] = "unknown"
            report["filled_cells"][col] = miss
        elif kind in {"date", "time", "id"}:
            drop_cols.append(col)                      # ما نخترع تاريخ ولا ساعة ولا رقم هوية
    if drop_cols:
        mask = df[drop_cols].isna().any(axis=1)
        if mask.mean() <= MAX_ROW_LOSS:
            report["dropped_rows_missing"] = int(mask.sum())
            df = df[~mask]
        else:                                          # الحذف راح ياكل الملف: نتركه ونعلّمه
            report["sparse_columns"].extend(drop_cols)

    df = df.reset_index(drop=True)
    report["rows_after"] = len(df)
    report["columns_after"] = len(df.columns)
    return df, report
# === نهاية التنظيف =========================================================

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

def run_analysis(df: pd.DataFrame, spec: dict, period: str = "M") -> dict | None:
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
    df = df.dropna(subset=[group_by])
    if df.empty:
        return None
    if kind == "monthly_trend":
        if not pd.api.types.is_datetime64_any_dtype(df[group_by]):
            return None
        per = df[group_by].dt.to_period(period)
        # أسبوعي: نعرض تاريخ بداية الأسبوع. يومي وشهري: الاسم الافتراضي للفترة
        keys = per.dt.start_time.dt.strftime("%Y-%m-%d") if period == "W" else per.astype(str)
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
        series = series.sort_index().tail(TREND_POINTS.get(period, 24))
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
Write in simple Arabic. Base every point ONLY on the numbers you were given; do not invent data. If a 'filter' field is present, the numbers cover only that filtered subset; mention this briefly in the summary.
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
        df = await asyncio.to_thread(read_file, file_path, file_type)
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
    df, cleaning_report = await asyncio.to_thread(clean_data, df, understanding)
    if df.empty or len(df.columns) < 2:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=MESSY_MESSAGE)
    # ---- Agent 2: خطة التحليل + التنفيذ باستخدام pandas ----
    plan = await ask_json(
        ANALYSIS_PROMPT,
        {"business_type": business_type, "profile": profile_data(df)},
    )
    specs = [s for s in plan.get("analyses", []) if isinstance(s, dict)][:MAX_ANALYSES]
    pairs = await asyncio.to_thread(lambda: [(s, run_analysis(df, s)) for s in specs])
    results = [r for _, r in pairs if r]
    used_specs = [s for s, r in pairs if r]
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
    # ---- حفظ البيانات النظيفة + الخطة حتى يكدر المستخدم يفلتر بدون إعادة رفع ----
    upload_id = Path(safe_filename).stem
    filter_meta = None
    try:
        cleanup_sessions()
        await asyncio.to_thread(save_session, upload_id, df, used_specs, business_type)
        filter_meta = await asyncio.to_thread(build_filter_meta, df)
    except Exception:
        logger.error("Could not save the filter session:\n%s", traceback.format_exc())

    db = SessionLocal()
    try:
        upload_record = Upload(
            filename=file.filename,
            status="uploaded",
        )
        db.add(upload_record)
        db.commit()
    except Exception as error:
        db.rollback()
        print("DATABASE SAVE ERROR:", repr(error))
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
        "upload_id": upload_id,
        "filter_meta": filter_meta,
    }

# ---------------------------------------------------------------------------
# الفلتر: نفس التحليل على بيانات المستخدم بعد فلتر يحدده بنفسه
# (بدون إعادة رفع وبدون إعادة تنظيف، نستعمل البيانات النظيفة المحفوظة بعد الرفع)
# ---------------------------------------------------------------------------
SESSIONS_DIR = Path("sessions")
SESSIONS_DIR.mkdir(exist_ok=True)
SESSION_TTL_SECONDS = 24 * 3600   # الجلسات الأقدم من 24 ساعة تنحذف
FILTER_MAX_VALUES = 200           # أقصى عدد قيم نعرضه لعمود تصنيفي بالواجهة
FILTER_MAX_COLUMNS = 60
MAX_FILTERS = 10
MAX_FILTER_VALUES = 500
PERIODS = {"day": "D", "week": "W", "month": "M"}
TREND_POINTS = {"D": 31, "W": 26, "M": 24}
SPEC_KEYS = ("kind", "group_by", "value", "agg", "top_n", "chart", "title")
WORDS = {
    "ar": {"only": "إظهار فقط", "hide": "إخفاء", "from": "من", "to": "إلى", "contains": "يحتوي",
           "not_contains": "لا يحتوي", "hidden_cols": "أعمدة مخفية", "gran": "التجميع الزمني",
           "day": "يومي", "week": "أسبوعي", "month": "شهري", "trend": "الاتجاه عبر الزمن",
           "no_rows": "ما اكو بيانات تطابق الفلتر. غيّر الشروط وجرّب مرة ثانية.",
           "no_analysis": "ما قدرنا نطلع تحليلاً بعد الفلتر. جرّب شروطاً أوسع أو لا تخفي أعمدة التحليل.",
           "expired": "انتهت صلاحية هذا التحليل (الجلسات تُحذف بعد 24 ساعة). ارفع الملف مرة ثانية."},
    "en": {"only": "only", "hide": "hide", "from": "from", "to": "to", "contains": "contains",
           "not_contains": "does not contain", "hidden_cols": "Hidden columns", "gran": "Time grouping",
           "day": "daily", "week": "weekly", "month": "monthly", "trend": "Trend over time",
           "no_rows": "No rows match the filter. Change the conditions and try again.",
           "no_analysis": "No analysis could be produced after filtering. Try wider conditions or unhide columns.",
           "expired": "This analysis has expired (sessions are deleted after 24 hours). Please upload the file again."},
}


class FilterSpec(BaseModel):
    column: str
    op: Literal["in", "not_in", "between", "contains", "not_contains"]
    values: list[str] = []
    value: str | None = None
    min: float | None = None
    max: float | None = None
    start: str | None = None
    end: str | None = None


class FilterRequest(BaseModel):
    upload_id: str
    filters: list[FilterSpec] = []
    hidden_columns: list[str] = []
    granularity: Literal["day", "week", "month"] | None = None
    date_column: str | None = None
    lang: str = "ar"


def save_session(upload_id: str, df: pd.DataFrame, specs: list, business_type: str) -> None:
    """نحفظ البيانات النظيفة + خطة التحليل حتى نكدر نفلترها بدون إعادة رفع (الاسم hex عشوائي نولّده نحن)."""
    SESSIONS_DIR.mkdir(exist_ok=True)
    df.to_pickle(SESSIONS_DIR / f"{upload_id}.pkl")
    meta = {"business_type": business_type, "specs": [{k: s.get(k) for k in SPEC_KEYS} for s in specs]}
    (SESSIONS_DIR / f"{upload_id}.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")


def load_session(upload_id: str, lang: str = "ar"):
    pkl, meta = SESSIONS_DIR / f"{upload_id}.pkl", SESSIONS_DIR / f"{upload_id}.json"
    if not pkl.exists() or not meta.exists():
        raise HTTPException(status_code=404, detail=WORDS["en" if lang == "en" else "ar"]["expired"])
    return pd.read_pickle(pkl), json.loads(meta.read_text(encoding="utf-8"))


def cleanup_sessions() -> None:
    cutoff = _time.time() - SESSION_TTL_SECONDS
    for p in SESSIONS_DIR.glob("*"):
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink()
        except OSError:
            pass


def build_filter_meta(df: pd.DataFrame) -> dict:
    """يوصف الأعمدة للواجهة حتى تبني خيارات الفلتر (تاريخ / رقم / تصنيف / نص)."""
    cols = []
    for c in list(df.columns)[:FILTER_MAX_COLUMNS]:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            d = s.dropna()
            if not d.empty:
                cols.append({"name": str(c), "kind": "date",
                             "min": d.min().strftime("%Y-%m-%d"), "max": d.max().strftime("%Y-%m-%d")})
        elif pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            d = s.dropna()
            if not d.empty and math.isfinite(float(d.min())) and math.isfinite(float(d.max())):
                cols.append({"name": str(c), "kind": "numeric", "min": float(d.min()), "max": float(d.max())})
        else:
            counts = s.astype(str).value_counts()
            if len(counts) <= FILTER_MAX_VALUES:
                cols.append({"name": str(c), "kind": "category", "values": [str(v) for v in counts.index]})
            else:
                cols.append({"name": str(c), "kind": "text"})
    return {
        "columns": cols,
        "date_columns": [c["name"] for c in cols if c["kind"] == "date"],
        "all_columns": [str(c) for c in df.columns],
    }


def _num_text(x) -> str:
    return f"{x:,.2f}".rstrip("0").rstrip(".")


def apply_filters(df: pd.DataFrame, filters: list, lang: str):
    """يطبق شروط المستخدم بـ pandas فقط (بدون eval). يرجع (البيانات بعد الفلتر, وصف الشروط)."""
    w = WORDS[lang]
    mask = pd.Series(True, index=df.index)
    descs = []
    for f in filters:
        if f.column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Unknown column: {f.column[:60]}")
        s = df[f.column]
        if f.op in ("in", "not_in"):
            vals = [str(v)[:200] for v in f.values[:MAX_FILTER_VALUES]]
            if not vals:
                continue
            m = s.astype(str).isin(set(vals))
            mask &= m if f.op == "in" else ~m
            shown = "، ".join(vals[:5]) + (" …" if len(vals) > 5 else "")
            descs.append(f"{f.column}: {w['only'] if f.op == 'in' else w['hide']} {shown}")
        elif f.op == "between":
            if pd.api.types.is_datetime64_any_dtype(s):
                start = pd.to_datetime(f.start, errors="coerce") if f.start else None
                end = pd.to_datetime(f.end, errors="coerce") if f.end else None
                if (f.start and pd.isna(start)) or (f.end and pd.isna(end)):
                    raise HTTPException(status_code=400, detail="Invalid date in filter.")
                if start is None and end is None:
                    continue
                tz = getattr(s.dt, "tz", None)
                if tz is not None:
                    start = start.tz_localize(tz) if start is not None else None
                    end = end.tz_localize(tz) if end is not None else None
                if start is not None:
                    mask &= s >= start
                if end is not None:
                    mask &= s < end + pd.Timedelta(days=1)       # يوم النهاية يدخل بالفلتر
                parts = [f"{w['from']} {f.start}" if f.start else "", f"{w['to']} {f.end}" if f.end else ""]
                descs.append(f"{f.column}: " + " ".join(p for p in parts if p))
            elif pd.api.types.is_numeric_dtype(s):
                if f.min is None and f.max is None:
                    continue
                if f.min is not None:
                    mask &= s >= f.min
                if f.max is not None:
                    mask &= s <= f.max
                parts = [f"{w['from']} {_num_text(f.min)}" if f.min is not None else "",
                         f"{w['to']} {_num_text(f.max)}" if f.max is not None else ""]
                descs.append(f"{f.column}: " + " ".join(p for p in parts if p))
            else:
                raise HTTPException(status_code=400, detail="'between' works only on numeric or date columns.")
        else:  # contains / not_contains
            text = (f.value or "").strip()[:200]
            if not text:
                continue
            m = s.astype(str).str.contains(text, case=False, regex=False, na=False)
            mask &= m if f.op == "contains" else ~m
            descs.append(f"{f.column} {w[f.op]} «{text}»")
    return df[mask], descs


def run_filtered_analysis(df: pd.DataFrame, saved: dict, req: "FilterRequest") -> dict:
    lang = "en" if req.lang == "en" else "ar"
    w = WORDS[lang]
    filtered, descs = apply_filters(df, req.filters, lang)
    if filtered.empty:
        raise HTTPException(status_code=422, detail=w["no_rows"])

    hidden = {h for h in req.hidden_columns if h in df.columns}
    specs = [dict(s) for s in saved.get("specs", [])]
    period = PERIODS.get(req.granularity or "")
    date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if req.date_column and req.date_column not in date_cols:
        raise HTTPException(status_code=400, detail="date_column is not a date column.")

    trend = [s for s in specs if s.get("kind") == "monthly_trend"]
    if req.date_column:
        for s in trend:
            s["group_by"] = req.date_column
    if period:
        label = w[req.granularity]
        for s in trend:
            s["title"] = f"{s.get('title') or w['trend']} ({label})"
        if not trend and date_cols:       # ما اكو مخطط زمني بالخطة الأصلية: نضيف واحد
            value = next((s["value"] for s in specs if s.get("value")), None)
            specs.append({"kind": "monthly_trend", "group_by": req.date_column or date_cols[0],
                          "value": value, "agg": "sum" if value else "count", "chart": "bar",
                          "title": f"{w['trend']} ({label})"})
        descs.append(f"{w['gran']}: {label}")
    if hidden:
        specs = [s for s in specs if s.get("group_by") not in hidden and s.get("value") not in hidden]
        descs.append(f"{w['hidden_cols']}: " + "، ".join(sorted(hidden)))

    results = [r for r in (run_analysis(filtered, s, period=period or "M") for s in specs) if r]
    if not results:
        raise HTTPException(status_code=422, detail=w["no_analysis"])
    return {
        "analyses": results,
        "filter_info": {"rows_before": len(df), "rows_after": len(filtered), "descriptions": descs},
    }


@app.post("/filter")
async def filter_analysis(req: FilterRequest):
    """يعيد نفس التحليلات على البيانات بعد الفلتر. الحسابات pandas، والذكاء الاصطناعي فقط يكتب الخلاصة الجديدة."""
    if not re.fullmatch(r"[0-9a-f]{32}", req.upload_id):
        raise HTTPException(status_code=400, detail="invalid upload id")
    if len(req.filters) > MAX_FILTERS or len(req.hidden_columns) > 100:
        raise HTTPException(status_code=400, detail="too many filters")
    df, saved = await asyncio.to_thread(load_session, req.upload_id, req.lang)
    out = await asyncio.to_thread(run_filtered_analysis, df, saved, req)
    try:
        ins = await ask_json(INSIGHTS_PROMPT, {
            "business_type": saved.get("business_type", ""),
            "filter": out["filter_info"]["descriptions"],
            "analyses": [{k: r[k] for k in ("title", "labels", "values")} for r in out["analyses"]],
        })
    except HTTPException:      # الذكاء الاصطناعي ما رد: نرجّع التحليل بدون خلاصة جديدة
        ins = None
    out.update({
        "summary": (ins or {}).get("summary", ""),
        "insights": (ins or {}).get("insights", []),
        "recommendations": (ins or {}).get("recommendations", []),
        "insights_error": ins is None,
    })
    return out


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
        "mean": "المتوسط", "median": "الوسيط", "mode": "الأكثر تكراراً", "zero": "صفر", "unknown": "غير محدد", "filtered_rows": "الصفوف بعد الفلتر",
    },
    "en": {
        "title": "Data Analysis Report", "file": "File", "date": "Date",
        "business": "Business type", "summary": "Summary",
        "rows": "Rows analyzed", "of": "of",
        "dups": "Duplicate rows removed", "missing": "Rows removed for missing data", "cols": "columns",
        "charts": "Charts", "insights": "What your data says", "recs": "What to do next",
        "filled": "Filled", "dropped": "Dropped column",
        "mean": "mean", "median": "median", "mode": "most common", "zero": "zero", "unknown": "unspecified", "filtered_rows": "Rows after filter",
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
        labels = [_clean(x, 80) for x in (a.get("labels") or [])][:31]
        vals = _nums(a.get("values"))[:31]
        n = min(len(labels), len(vals))
        labels, vals = labels[:n], vals[:n]
        if not n:
            return
        is_pie = a.get("chart") == "pie"
        is_time = n > 1 and all(re.fullmatch(r"\d{4}-\d{2}(?:-\d{2})?", l) for l in labels)
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
                p_ = lab.split("-")
                short = f"{p_[2]}/{p_[1]}" if len(p_) == 3 else f"{p_[1]}/{p_[0][2:]}"
                self.put(short, bx + bw / 2, base_y - 12, 7, GRAY, "c")
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

    fi = data.get("filter_info")
    if isinstance(fi, dict):
        items = [(f"{t['filtered_rows']}: {fmt(fi.get('rows_after'))} {t['of']} {fmt(fi.get('rows_before'))}", False)]
        items += [(_clean(d, 120), False) for d in (fi.get("descriptions") or [])[:12]]
        r.chips(items)

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
    except Exception as error:  # أي خطأ ثاني: نطبعه باللوغ ونرجّع سببه للواجهة@
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

