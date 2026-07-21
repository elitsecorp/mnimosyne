"""SQLAlchemy ORM models for all database tables."""

from __future__ import annotations

import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, LargeBinary, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""


class ChatSession(Base):
    """Chat session grouping conversation messages."""

    __tablename__ = "chat_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(256), nullable=False, default="New Chat")
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=lambda: datetime.datetime.now(datetime.UTC))
    updated_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=lambda: datetime.datetime.now(datetime.UTC), onupdate=lambda: datetime.datetime.now(datetime.UTC))

    messages: Mapped[list[Message]] = relationship("Message", back_populates="session", order_by="Message.timestamp")


class Message(Base):
    """Stores conversation messages (user and assistant)."""

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("chat_sessions.id"), nullable=True)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[datetime.datetime] = mapped_column(DateTime, default=lambda: datetime.datetime.now(datetime.UTC))

    session: Mapped[ChatSession] = relationship("ChatSession", back_populates="messages")
    embeddings: Mapped[list[Embedding]] = relationship("Embedding", back_populates="message")


class Embedding(Base):
    """Stores vector embeddings linked to messages."""

    __tablename__ = "embeddings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    message_id: Mapped[int] = mapped_column(Integer, ForeignKey("messages.id"), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)

    message: Mapped[Message] = relationship("Message", back_populates="embeddings")


class Entity(Base):
    """Named entities extracted from conversations."""

    __tablename__ = "entities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False, unique=True)
    type: Mapped[str] = mapped_column(String(64), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)


class Relationship(Base):
    """Relationships between entities."""

    __tablename__ = "relationships"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    subject: Mapped[str] = mapped_column(String(256), nullable=False)
    predicate: Mapped[str] = mapped_column(String(128), nullable=False)
    object: Mapped[str] = mapped_column(String(256), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    is_owner: Mapped[bool] = mapped_column(Integer, default=0)


class Fact(Base):
    """Atomic facts extracted from conversations."""

    __tablename__ = "facts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    subject: Mapped[str] = mapped_column(String(256), nullable=False)
    predicate: Mapped[str] = mapped_column(String(128), nullable=False)
    object: Mapped[str] = mapped_column(String(256), nullable=False)
    source_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    timestamp: Mapped[datetime.datetime] = mapped_column(DateTime, default=lambda: datetime.datetime.now(datetime.UTC))
