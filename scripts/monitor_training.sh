#!/bin/bash
# Training monitoring script
# Usage: bash scripts/monitor_training.sh [interval_seconds]

INTERVAL="${1:-60}"  # Default 60 seconds
INSTANCE="instance-20251202-055916"
ZONE="northamerica-northeast1-b"
PROJECT="one-shot-rlvr-cs229"

echo "=== Training Monitor ==="
echo "Refresh interval: ${INTERVAL}s"
echo "Press Ctrl+C to stop"
echo ""

while true; do
    /Users/gil/google-cloud-sdk/bin/gcloud compute ssh $INSTANCE \
        --project=$PROJECT \
        --zone=$ZONE \
        --command="
        clear
        echo '╔═══════════════════════════════════════════════════════════╗'
        echo '║           Training Monitor - '\$(date '+%H:%M:%S')'                    ║'
        echo '╚═══════════════════════════════════════════════════════════╝'
        echo ''
        
        # Steps
        STEP_COUNT=\$(grep -E 'step:[0-9]+' ~/qwen0.5b_log.txt 2>/dev/null | wc -l)
        echo \"📊 Steps: \$STEP_COUNT\"
        if [ \$STEP_COUNT -gt 0 ]; then
            LAST_STEP=\$(grep -E 'step:[0-9]+' ~/qwen0.5b_log.txt 2>/dev/null | tail -1)
            STEP_TIME=\$(echo \"\$LAST_STEP\" | grep -oP 'timing_s/step:\K[0-9.]+' || echo 'N/A')
            echo \"   Last step time: \${STEP_TIME}s\"
            
            # Rewards
            REWARD=\$(echo \"\$LAST_STEP\" | grep -oP 'critic/rewards/mean:\K[0-9.]+' || echo 'N/A')
            SCORE=\$(echo \"\$LAST_STEP\" | grep -oP 'critic/score/mean:\K[0-9.]+' || echo 'N/A')
            echo \"   Reward mean: \$REWARD | Score mean: \$SCORE\"
        fi
        echo ''
        
        # GPU
        nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits | \
            awk -F', ' '{printf \"🔥 GPU: %d/%d MB (%.1f%%) | Util: %d%% | Temp: %d°C\n\", \$1, \$2, (\$1/\$2*100), \$3, \$4}'
        echo ''
        
        # Process
        if ps aux | grep 'main_ppo' | grep -v grep > /dev/null; then
            echo '✅ Process: RUNNING'
        else
            echo '❌ Process: STOPPED'
        fi
        echo ''
        
        # Recent steps
        if [ \$STEP_COUNT -gt 0 ]; then
            echo 'Recent steps:'
            grep -E 'step:[0-9]+' ~/qwen0.5b_log.txt 2>/dev/null | tail -3 | \
                sed 's/.*step:\([0-9]\+\).*/  Step \1/' || echo '  (parsing...)'
        fi
        " 2>&1 | grep -v "Python 3.9"
    
    echo ""
    echo "Refreshing in ${INTERVAL}s... (Ctrl+C to stop)"
    sleep $INTERVAL
done

