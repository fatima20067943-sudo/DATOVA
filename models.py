
from sqlalchemy import Column, Integer, String, DateTime, text
from database import Base


class Upload(Base):
    __tablename__ = "uploads"

    id = Column(Integer, primary_key=True, autoincrement=True)

    filename = Column(String(255), nullable=False)

    uploaded_at = Column(
        DateTime,
        server_default=text("CURRENT_TIMESTAMP"),
        nullable=False,
    )

    status = Column(
        String(50),
        nullable=False,
        default="uploaded",
    )
