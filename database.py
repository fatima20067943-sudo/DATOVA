
import os

from dotenv import load_dotenv
from sqlalchemy import URL, create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()

# قراءة معلومات الاتصال من ملف البيئة
db_url = URL.create(
    drivername="mysql+pymysql",
    username=os.getenv("MYSQLUSER"),
    password=os.getenv("MYSQLPASSWORD"),
    host=os.getenv("MYSQLHOST"),
    port=int(os.getenv("MYSQLPORT", "3306")),
    database=os.getenv("MYSQLDATABASE"),
)

# إنشاء الاتصال بقاعدة البيانات
engine = create_engine(
    db_url,
    pool_pre_ping=True,
    pool_recycle=3600,
)

# إنشاء جلسة للتعامل مع قاعدة البيانات
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)

# الأساس لإنشاء جداول قاعدة البيانات لاحقًا
Base = declarative_base()


# فحص الاتصال
def test_connection():
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
        return True
