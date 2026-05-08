"""Supabase persistence layer for the violation pipeline (Phase 2C.2)."""

from backend.storage.supabase_client import (
    EventUploader,
    SiteManager,
    SupabaseClient,
    WorkerRegistry,
)

__all__ = ["SupabaseClient", "SiteManager", "WorkerRegistry", "EventUploader"]
