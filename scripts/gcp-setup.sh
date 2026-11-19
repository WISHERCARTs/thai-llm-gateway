#!/usr/bin/env bash
# One-time Google Cloud setup for thai-llm-gateway. Run it in Cloud Shell, in the gateway's project:
#
#   gcloud config set project YOUR_PROJECT_ID
#   curl -fsSL https://raw.githubusercontent.com/WISHERCARTs/thai-llm-gateway/main/scripts/gcp-setup.sh -o gcp-setup.sh
#   bash gcp-setup.sh
#
# It is safe to run again: every step checks what already exists.
# Secrets are typed into hidden prompts and go straight to Secret Manager; nothing is printed or saved to disk.
set -euo pipefail

REGION="${REGION:-asia-southeast1}"
GITHUB_REPO="${GITHUB_REPO:-WISHERCARTs/thai-llm-gateway}"
SERVICE="thai-llm-gateway"
AR_REPO="gateway"
RUNTIME_SA_NAME="gateway-runtime"
DEPLOY_SA_NAME="gateway-deployer"
POOL="github"
PROVIDER="github"

PROJECT_ID="$(gcloud config get-value project 2>/dev/null)"
if [[ -z "$PROJECT_ID" ]]; then
  echo "Set the project first: gcloud config set project YOUR_PROJECT_ID" >&2
  exit 1
fi
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
RUNTIME_SA="${RUNTIME_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
DEPLOY_SA="${DEPLOY_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

step() { printf '\n== %s\n' "$*"; }

step "Project ${PROJECT_ID} (${PROJECT_NUMBER}), region ${REGION}"
if [[ "$(gcloud billing projects describe "$PROJECT_ID" --format='value(billingEnabled)' 2>/dev/null)" != "True" ]]; then
  echo "Billing is not enabled on this project. Cloud Run needs a billing account even on the free tier:" >&2
  echo "https://console.cloud.google.com/billing/linkedaccount?project=${PROJECT_ID}" >&2
  exit 1
fi

step "Enabling APIs (takes a minute the first time)"
gcloud services enable run.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com \
  iamcredentials.googleapis.com sts.googleapis.com --quiet

step "Artifact Registry repository for the image"
gcloud artifacts repositories describe "$AR_REPO" --location "$REGION" >/dev/null 2>&1 ||
  gcloud artifacts repositories create "$AR_REPO" --location "$REGION" --repository-format docker \
    --description "thai-llm-gateway images" --quiet

step "Service accounts"
for sa in "$RUNTIME_SA_NAME:the Cloud Run service" "$DEPLOY_SA_NAME:GitHub Actions deploys"; do
  name="${sa%%:*}"
  gcloud iam service-accounts describe "${name}@${PROJECT_ID}.iam.gserviceaccount.com" >/dev/null 2>&1 ||
    gcloud iam service-accounts create "$name" --display-name "thai-llm-gateway: ${sa#*:}" --quiet
done
for role in roles/run.admin roles/artifactregistry.writer; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" --member "serviceAccount:${DEPLOY_SA}" --role "$role" \
    --condition None --quiet >/dev/null
done
# The deployer may run the service as the runtime account, and nothing else.
gcloud iam service-accounts add-iam-policy-binding "$RUNTIME_SA" --member "serviceAccount:${DEPLOY_SA}" \
  --role roles/iam.serviceAccountUser --quiet >/dev/null

step "Workload Identity Federation: GitHub Actions in ${GITHUB_REPO} only, no JSON keys"
gcloud iam workload-identity-pools describe "$POOL" --location global >/dev/null 2>&1 ||
  gcloud iam workload-identity-pools create "$POOL" --location global --display-name "GitHub Actions" --quiet
gcloud iam workload-identity-pools providers describe "$PROVIDER" --location global \
  --workload-identity-pool "$POOL" >/dev/null 2>&1 ||
  gcloud iam workload-identity-pools providers create-oidc "$PROVIDER" --location global \
    --workload-identity-pool "$POOL" --display-name "GitHub" \
    --issuer-uri "https://token.actions.githubusercontent.com" \
    --attribute-mapping "google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.ref=assertion.ref" \
    --attribute-condition "assertion.repository == '${GITHUB_REPO}'" --quiet
POOL_ID="projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL}"
gcloud iam service-accounts add-iam-policy-binding "$DEPLOY_SA" --role roles/iam.workloadIdentityUser \
  --member "principalSet://iam.googleapis.com/${POOL_ID}/attribute.repository/${GITHUB_REPO}" --quiet >/dev/null

step "Secrets"
secret_exists() { gcloud secrets describe "$1" >/dev/null 2>&1; }
put_secret() { # name, value on stdin
  secret_exists "$1" || gcloud secrets create "$1" --replication-policy automatic --quiet >/dev/null
  gcloud secrets versions add "$1" --data-file - --quiet >/dev/null
  gcloud secrets add-iam-policy-binding "$1" --member "serviceAccount:${RUNTIME_SA}" \
    --role roles/secretmanager.secretAccessor --quiet >/dev/null
  echo "  saved $1"
}
ask_secret() { # name, prompt
  if secret_exists "$1" && [[ "${RESET:-}" != *"$1"* ]]; then
    echo "  $1 already set (run with RESET=$1 to replace it)"
    return
  fi
  local value=""
  while [[ -z "$value" ]]; do
    read -r -s -p "  $2: " value
    echo
  done
  printf '%s' "$value" | put_secret "$1"
}
random() { python3 -c 'import secrets; print(secrets.token_urlsafe(32), end="")'; }

ask_secret GEMINI_API_KEY "Gemini API key (from aistudio.google.com/apikey, in this project)"
ask_secret UPSTASH_REDIS_REST_URL "Upstash Redis REST URL (https://...upstash.io)"
ask_secret UPSTASH_REDIS_REST_TOKEN "Upstash Redis REST token"
ask_secret DATABASE_URL "Neon pooled connection string (postgresql://...-pooler...)"
ask_secret LANGFUSE_PUBLIC_KEY "Langfuse public key (pk-lf-...)"
ask_secret LANGFUSE_SECRET_KEY "Langfuse secret key (sk-lf-...)"
# Generated here and never shown: the agent service and the portfolio read their client key from Secret Manager.
secret_exists USER_HASH_SALT || random | put_secret USER_HASH_SALT
secret_exists GATEWAY_CLIENT_KEYS ||
  printf 'thai-rag-agents:%s,portfolio:%s,smoke-test:%s' "$(random)" "$(random)" "$(random)" | put_secret GATEWAY_CLIENT_KEYS

step "Database schema (usage_log)"
if command -v psql >/dev/null; then
  MIGRATION="20261119000000_usage_log.sql"
  curl -fsSL "https://raw.githubusercontent.com/${GITHUB_REPO}/main/db/migrations/${MIGRATION}" -o "/tmp/${MIGRATION}"
  DB_URL="$(gcloud secrets versions access latest --secret DATABASE_URL)"
  psql "$DB_URL" -v ON_ERROR_STOP=1 -q -f "/tmp/${MIGRATION}" -c "
    create table if not exists public.schema_migrations (name text primary key, applied_at timestamptz not null default now());
    insert into public.schema_migrations (name) values ('${MIGRATION}') on conflict do nothing;"
  unset DB_URL
  echo "  applied ${MIGRATION}"
else
  echo "  psql not found; run: DATABASE_URL=... python -m gateway.migrate"
fi

step "Done. Send these three lines to Claude (they are not secret):"
echo "GCP_PROJECT_ID=${PROJECT_ID}"
echo "GCP_WIF_PROVIDER=${POOL_ID}/providers/${PROVIDER}"
echo "GCP_DEPLOY_SA=${DEPLOY_SA}"
