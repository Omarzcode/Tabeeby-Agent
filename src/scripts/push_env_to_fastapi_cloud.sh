#!/usr/bin/env bash
# Push every non-empty KEY=VALUE from .env to FastAPI Cloud.
# Keys/tokens/secrets are stored as encrypted secrets; everything else as plain vars.
# Run from src/ after `fastapi login` and a first `fastapi deploy`.
#   bash scripts/push_env_to_fastapi_cloud.sh            # uses .env
#   bash scripts/push_env_to_fastapi_cloud.sh .env.prod  # another file
set -euo pipefail
ENV_FILE="${1:-.env}"
[ -f "$ENV_FILE" ] || { echo "No $ENV_FILE found"; exit 1; }

# Values that only make sense locally are skipped.
SKIP_RE='^(OLLAMA_API_URL|VECTOR_DB_PATH)$'

while IFS= read -r line || [ -n "$line" ]; do
  line="${line%%$'\r'}"
  [[ -z "$line" || "$line" =~ ^[[:space:]]*# || "$line" != *=* ]] && continue
  key="$(echo "${line%%=*}" | xargs)"
  value="$(echo "${line#*=}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/^"\(.*\)"$/\1/' -e "s/^'\(.*\)'$/\1/")"
  [[ -z "$value" || "$key" =~ $SKIP_RE ]] && continue
  if [[ "$key" =~ (KEY|TOKEN|SECRET|PASSWORD)$ ]]; then
    echo "secret  $key"
    fastapi cloud env set --secret "$key" "$value" >/dev/null
  else
    echo "var     $key=$value"
    fastapi cloud env set "$key" "$value" >/dev/null
  fi
done < "$ENV_FILE"

echo
echo "Done. Env changes do not redeploy automatically: run 'fastapi deploy' now."
