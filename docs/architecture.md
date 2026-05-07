# İşGözcü - System Architecture

> Architecture documentation will be expanded in later phases.

## Overview

İşGözcü is a construction site safety monitoring system that uses computer vision to detect PPE (Personal Protective Equipment) violations in real time.

## High-Level Components

- **Inference Engine**: YOLOv8 + ByteTrack for detection and multi-object tracking
- **Backend API**: FastAPI service exposing detection, reporting, and snapshot endpoints
- **Database**: Supabase (PostgreSQL) for events, sessions, and report metadata
- **Frontend**: Next.js dashboard for live view, reports, and incident review
- **Storage**: Snapshot images for violation evidence (KVKK-compliant)

## Detection Pipeline

1. Video stream / file ingestion
2. Frame extraction
3. YOLO inference (PPE classes + person)
4. ByteTrack association across frames
5. Violation rules engine (helmet missing, vest missing, restricted zone)
6. Debounce by `MIN_VIOLATION_DURATION_SECONDS`
7. Persist event + snapshot to Supabase

## Compliance & Privacy

- KVKK-compliant: workers tracked anonymously by track ID, not biometrics
- Faces blurred in stored snapshots (Phase 2+)
- Audit log for all report exports
