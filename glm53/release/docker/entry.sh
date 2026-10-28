#!/bin/sh
# entry.sh: the image's commands
#   install DIR          copy the release's files (/opt/glm53) to DIR, a directory mounted from the host: the same
#                        files as the unpacked archive, for tf-glm53 and the launcher on the host
#   serve LOG ARGS...    the server (bin/tensorfold-native serve ARGS), its output to LOG; LOG - : to stdout (tf-glm53
#                        starts the ranks so, the output into the node's log keeper)
#   version              the release's VERSION
set -e
case "${1:-help}" in
  install)
    [ -n "$2" ] && [ -d "$2" ] || { echo "install DIR: a directory mounted from the host (-v \"\$PWD\":/out ... install /out)"; exit 2; }
    # a directory Docker made for the mount is root's: the user (--user) cannot write it
    [ -w "$2" ] || { echo "install: $2 is not writable as uid $(id -u) (make the directory before docker run: mkdir glm53)"; exit 2; }
    cp -a /opt/glm53/. "$2"/ && echo "installed $(cat /opt/glm53/VERSION) in $2" ;;
  serve)
    [ -n "$2" ] || { echo "serve LOG ARGS..."; exit 2; }
    log=$2; shift 2
    [ "$log" = - ] && exec /opt/glm53/bin/tensorfold-native serve "$@" 2>&1
    exec /opt/glm53/bin/tensorfold-native serve "$@" > "$log" 2>&1 ;;
  version) cat /opt/glm53/VERSION ;;
  *) sed -n '2,7p' "$0" | sed 's/^# \{0,1\}//'; [ "$1" = help ] || exit 2 ;;
esac
