#!/bin/bash
# check-deploy.sh - definitive deploy check for a commit (default: HEAD).
# Reports the workflow conclusion, any failed steps, and whether the live
# site is actually serving that commit (via the SHA stamp in the page).
#
# Usage: scripts/check-deploy.sh [sha]
set -u
REPO="OmarAlterkait/OmarAlterkait.github.io"
SITE="https://omaralterkait.github.io/"
SHA=$(git rev-parse "${1:-HEAD}")

echo "Commit:   $SHA"

RUN=$(curl -s "https://api.github.com/repos/$REPO/actions/runs?head_sha=$SHA" | python3 -c "
import json, sys
d = json.load(sys.stdin)
runs = [r for r in d.get('workflow_runs', []) if r['name'] == 'Deploy site to Pages']
if not runs:
    print('none - -')
else:
    r = runs[0]
    print(r['id'], r['status'], r['conclusion'] or 'pending')
")
read -r RUN_ID RUN_STATUS RUN_CONCLUSION <<< "$RUN"
echo "Workflow: status=$RUN_STATUS conclusion=$RUN_CONCLUSION"

if [ "$RUN_CONCLUSION" = "failure" ]; then
    echo "Failed steps:"
    curl -s "https://api.github.com/repos/$REPO/actions/runs/$RUN_ID/jobs" | python3 -c "
import json, sys
for j in json.load(sys.stdin)['jobs']:
    for s in j['steps']:
        if s['conclusion'] not in ('success', 'skipped', None):
            print('   ', j['name'], '/', s['name'], '->', s['conclusion'])
"
fi

if curl -fsSL "$SITE?v=$SHA" 2>/dev/null | grep -q "deployed: $SHA"; then
    echo "LIVE:     site is serving this commit"
    exit 0
else
    echo "NOT LIVE: site is not (yet) serving this commit"
    exit 1
fi
