#!/usr/bin/env bash
# Show Q&A samples + live dataset stats.
#
#   qna.sh            latest sample, full (question + reasoning + answer)
#   qna.sh 5          sample #5, full
#   qna.sh -s [N]     question + answer only (reasoning hidden)
#   qna.sh -m [N]     metadata block (quality scores)
#   qna.sh -          index of all samples
#   qna.sh -t         topic tally
#   qna.sh -stats     counters: samples, questions, topics, queues, rate
#
# Dataset: /root/datasets/raw_001.txt   (blocks separated by a line of 50+ '=')

FILE="${QNA_FILE:-/root/datasets/raw_001.txt}"
LOG_DIR="${QNA_LOG_DIR:-/root/logs}"

# mawk lacks ERE intervals, so separators match "line of = signs, 50+ long".
count() { grep -c '^TOPIC:' "$FILE" 2>/dev/null || echo 0; }

block() {
  awk -v want="$1" '
    /^=+$/ && length($0) >= 50 { n++; show = (n == want); next }
    show { print }
  ' "$FILE"
}

strip_reasoning() {
  awk '
    /^<think>$/   { inthink=1; next }
    /^<\/think>$/ { inthink=0; next }
    inthink { next }
    { print }
  '
}

stats() {
  local samples questions topics qchars rchars achars scores L first last mins rate s
  samples=$(count)
  questions=$(grep -c '^QUESTION:' "$FILE" 2>/dev/null || echo 0)
  topics=$(grep '^TOPIC:' "$FILE" 2>/dev/null | sort -u | wc -l)

  qchars=$(awk '/^QUESTION:/{getline; print length($0)}' "$FILE" | awk '{s+=$1;n++} END{if(n) printf "%d", s/n; else print 0}')
  rchars=$(awk '/^<think>$/{f=1;next} /^<\/think>$/{f=0} f{n+=length($0)+1} END{print n+0}' "$FILE")
  achars=$(awk '/^ANSWER:/{f=1;next} /^METADATA:/{f=0} f{n+=length($0)+1} END{print n+0}' "$FILE")
  scores=$(grep '^QUALITY_SCORE:' "$FILE" 2>/dev/null | awk '{gsub("QUALITY_SCORE: ","");s+=$1;n++} END{if(n) printf "%.3f", s/n; else print "-"}')
  s=$(( samples > 0 ? samples : 1 ))

  echo "┌─ DATASET ──────────────────────────────────────────"
  printf "│ samples (accepted Q&A)   %s\n" "$samples"
  printf "│ questions asked          %s\n" "$questions"
  printf "│ topics                   %s unique\n" "$topics"
  printf "│ avg question             %s chars\n" "$qchars"
  printf "│ avg reasoning            %d chars\n" "$(( rchars / s ))"
  printf "│ avg answer               %d chars\n" "$(( achars / s ))"
  printf "│ avg quality score        %s\n" "$scores"
  printf "│ file                     %s\n" "$(du -h "$FILE" 2>/dev/null | cut -f1)"

  echo "├─ PIPELINE ─────────────────────────────────────────"
  L=$(ls -t "$LOG_DIR"/distill_*.log 2>/dev/null | head -1)
  if [ -n "$L" ]; then
    printf "│ accepted (this log)      %s\n" "$(grep -c 'ACCEPTED and saved' "$L")"
    printf "│ rejected (this log)      %s\n" "$(grep -c 'REJECTED' "$L")"
    printf "│ topics retired           %s\n" "$(grep -c 'Topic retired' "$L")"
    printf "│ judge failures           %s\n" "$(grep -c 'judge output unparsable' "$L")"
    first=$(grep -m1 'System running' "$L" | awk '{print $2}')
    last=$(tail -1 "$L" | awk '{print $2}')
    if [ -n "$first" ] && [ -n "$last" ]; then
      mins=$(awk -v a="$first" -v b="$last" 'BEGIN{split(a,x,":");split(b,y,":");d=(y[1]*3600+y[2]*60+y[3])-(x[1]*3600+x[2]*60+x[3]);if(d<0)d+=86400;printf "%d",d/60}')
      if [ "${mins:-0}" -gt 0 ]; then
        rate=$(awk -v sm="$samples" -v m="$mins" 'BEGIN{printf "%.1f", sm/m}')
        printf "│ rate                     %s samples/min (%s min)\n" "$rate" "$mins"
      fi
    fi
  fi

  echo "├─ QUEUES ───────────────────────────────────────────"
  printf "│ topic pool               %s\n" "$(redis-cli llen distill:topic_pool 2>/dev/null || echo n/a)"
  printf "│ answer backlog           %s\n" "$(redis-cli llen distill:queue:answer 2>/dev/null || echo n/a)"
  printf "│ question backlog         %s\n" "$(redis-cli llen distill:queue:question 2>/dev/null || echo n/a)"
  printf "│ quality backlog          %s\n" "$(redis-cli llen distill:queue:quality 2>/dev/null || echo n/a)"
  printf "│ service                  %s\n" "$(systemctl is-active distillation 2>/dev/null || echo n/a)"
  echo "└────────────────────────────────────────────────────"

  echo
  echo "TOPICS:"
  grep '^TOPIC:' "$FILE" 2>/dev/null | sed 's/^TOPIC: //' | sort | uniq -c | sort -rn | sed 's/^/  /'
}

case "$1" in
  ""|--last)
    n=$(count); [ "$n" -eq 0 ] && { echo "no samples yet in $FILE"; exit 0; }
    echo "### sample $n / $n"; block "$n" ;;
  -s|--short)
    n="${2:-$(count)}"; [ "$n" -eq 0 ] && { echo "no samples yet in $FILE"; exit 0; }
    echo "### sample $n / $(count)   (question + answer, reasoning hidden)"
    block "$n" | strip_reasoning ;;
  -m|--meta)
    n="${2:-$(count)}"; echo "### sample $n / $(count)   (metadata)"
    block "$n" | awk '/^METADATA:/{f=1} f' ;;
  -t|--topics)
    echo "### $(count) samples across $(grep '^TOPIC:' "$FILE" 2>/dev/null | sort -u | wc -l) topics"
    grep '^TOPIC:' "$FILE" 2>/dev/null | sed 's/^TOPIC: //' | sort | uniq -c | sort -rn ;;
  -stats|--stats|-S)
    stats ;;
  -|--index)
    echo "### $(count) samples"
    awk '
      /^=+$/ && length($0) >= 50 { n++; next }
      /^TOPIC: /    { t=substr($0,8) }
      /^QUESTION:/  { getline q; printf "%3d  %-11s %s\n", n, substr(t,1,40), substr(q,1,95) }
    ' "$FILE" ;;
  -h|--help)
    sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//' ;;
  *[0-9]*)
    echo "### sample $1 / $(count)"; block "$1" ;;
  *)
    echo "usage: qna.sh [N] | -s [N] | -m [N] | -stats | - | -t | -h" ;;
esac
