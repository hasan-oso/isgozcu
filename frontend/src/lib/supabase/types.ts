// TypeScript shapes mirroring the Supabase schema (Phase 2C.1 + 2C.2).
// snake_case to match Postgres columns directly.

export interface Site {
  id: string;
  name: string;
  location: string | null;
  created_at?: string;
}

export interface Camera {
  id: string;
  site_id: string;
  name: string;
  created_at?: string;
}

export interface Worker {
  id: string;
  site_id: string;
  tracker_id: string;
  first_seen: string | null;
  last_seen: string | null;
  total_violations: number;
  created_at?: string;
}

export type ViolationType = "no_hardhat" | "no_vest" | "no_mask";

export interface ViolationEvent {
  id: string;
  camera_id: string;
  worker_id: string;
  violation_type: ViolationType | string;
  violation_label_tr: string;
  started_at: string;
  ended_at: string | null;
  duration_seconds: number;
  snapshot_url: string | null;
  confidence_avg: number;
  is_active: boolean;
  metadata: Record<string, unknown> | null;
  created_at: string;
}

// Embedded Worker on a joined fetch (events?select=*,workers(tracker_id))
export interface ViolationEventWithWorker extends ViolationEvent {
  workers?: Pick<Worker, "tracker_id"> | null;
}

// Generic Database type for supabase-js client typing.
export interface Database {
  public: {
    Tables: {
      sites: { Row: Site; Insert: Partial<Site>; Update: Partial<Site> };
      cameras: { Row: Camera; Insert: Partial<Camera>; Update: Partial<Camera> };
      workers: { Row: Worker; Insert: Partial<Worker>; Update: Partial<Worker> };
      events: {
        Row: ViolationEvent;
        Insert: Partial<ViolationEvent>;
        Update: Partial<ViolationEvent>;
      };
    };
  };
}
