-- One row per gateway request, written by the gateway. Plain Postgres: runs on Neon, Supabase or local.
create table if not exists public.usage_log (
  id bigint generated always as identity primary key,
  created_at timestamptz not null default now(),
  request_id text not null,
  client text not null,
  end_user_hash text,
  provider text,
  model text,
  status int not null,
  input_tokens int not null default 0,
  output_tokens int not null default 0,
  cost_usd numeric(12, 8) not null default 0,
  latency_ms int not null default 0,
  pii_redactions int not null default 0,
  injection_score real not null default 0,
  blocked_reason text,
  cache_hit boolean not null default false
);

create index if not exists usage_log_created_at_idx on public.usage_log (created_at desc);
create index if not exists usage_log_client_idx on public.usage_log (client, created_at desc);

-- Daily cost per client and model, for the README report and a dashboard.
create or replace view public.usage_daily as
select date_trunc('day', created_at) as day,
       client,
       model,
       count(*) as requests,
       count(*) filter (where blocked_reason is not null) as blocked,
       sum(input_tokens) as input_tokens,
       sum(output_tokens) as output_tokens,
       sum(cost_usd) as cost_usd,
       percentile_cont(0.95) within group (order by latency_ms) as p95_latency_ms
from public.usage_log
group by 1, 2, 3;
