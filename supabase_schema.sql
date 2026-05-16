-- Run this in your Supabase SQL Editor

create table if not exists public.sessions (
  id uuid default gen_random_uuid() primary key,
  user_id uuid references auth.users(id) on delete cascade not null,
  label text not null default 'Untitled',
  mode text not null check (mode in ('timed','continuous')),
  planned_duration integer,
  actual_duration integer not null default 0,
  alert_count integer not null default 0,
  avg_ear float not null default 0,
  min_ear float not null default 1,
  max_drowsy_episode float not null default 0,
  alert_timestamps jsonb default '[]'::jsonb,
  created_at timestamptz default now()
);

alter table public.sessions enable row level security;

create policy "Users can view own sessions"
  on public.sessions for select using (auth.uid() = user_id);

create policy "Users can insert own sessions"
  on public.sessions for insert with check (auth.uid() = user_id);

create policy "Users can delete own sessions"
  on public.sessions for delete using (auth.uid() = user_id);
