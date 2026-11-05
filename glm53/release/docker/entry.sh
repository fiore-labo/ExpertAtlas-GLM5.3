#!/bin/sh
# entry.sh: the image's commands
#   up                   from this image to a running engine: on the first node,
#                          docker run --rm --network host -e USER="$USER" -v "$HOME":/host IMAGE up
#                        copies the release's files to ~/expertatlas on the host, writes its glm53.conf (RUNTIME=docker),
#                        and runs tf-glm53 quickstart there over ssh to the host: the launcher (the dashboard and the
#                        control API) runs on the host and stays; the ranks run in containers of this image. No API key;
#                        both open to the LAN. What it cannot do itself it says: STOP, FIX, THEN. This container exits then.
#                        The checkpoint and the 3-bit bank are always needed; every expert loads at 3 bits unless BITS says.
#                        Every option is -e NAME=VALUE (QUICKSTART.md lists them):
#                          where the files are: WEIGHTS, BANK_3, BANK_2, BANK_4, BANK_5, DRAFTER (DFlash2's model)
#                          the start: BITS=2..5, ALLOC=NAME (a saved allocation), NODES (the workers on a switch)
#                          any key of the engine settings (CONFIG.md): CONTEXT_LENGTH, KV_SIZE (a number or max), KV,
#                          DRAFTER_KIND, MULTI, MULTI_WAIT, RESERVE_GB, MAX_TOKENS, THINKING_SHARE, NAME, HOST, ...
#                          here: EA_DIR=NAME (under the home, default expertatlas), PORT (the API),
#                          LAUNCHER_PORT (default PORT + 10), IMAGE (what the nodes pull, default this image)
#   install DIR          copy the release's files (/opt/glm53) to DIR, a directory mounted from the host: the same
#                        files as the unpacked archive, for tf-glm53 and the launcher on the host
#   serve LOG ARGS...    the server (bin/tensorfold-native serve ARGS), its output to LOG; LOG - : to stdout (tf-glm53
#                        starts the ranks so, the output into the node's log keeper)
#   version              the release's VERSION
set -e
UP_LINE='docker run --rm --network host -e USER="$USER" -v "$HOME":/host IMAGE up'
# the environment that is not an option: the container's own, and USER
UP_OWN="PATH HOME HOSTNAME PWD OLDPWD SHLVL TERM USER _ LANG LC_ALL container TF_NCCL_LIB NVIDIA_DRIVER_CAPABILITIES NVIDIA_VISIBLE_DEVICES"
# the options this script takes itself, before the launcher starts
UP_HERE="EA_DIR IMAGE PORT LAUNCHER_PORT"
# one word for the remote shell
q() { printf "'%s'" "$(printf %s "$1" | sed "s/'/'\\\\''/g")"; }
cmd_up() {
  dir=${EA_DIR:-expertatlas}
  [ -d /host ] && [ -n "${USER:-}" ] && [ "${USER:-}" != root -o -d /host/.ssh ] || {
    echo "STOP: up needs the host's home at /host and your user name in USER."
    echo "FIX:  run it on the first node, as your user, exactly so: $UP_LINE"
    exit 2; }
  case $dir in */*|.*|'') echo "STOP: EA_DIR=$dir: one directory name under the home"; exit 2 ;; esac
  [ "$(id -u)" = 0 ] || { echo "STOP: up runs as root in the container (it writes in /host as its owner): drop --user"; exit 2; }
  for v in PORT LAUNCHER_PORT; do
    eval "x=\${$v:-}"
    case $x in ''|*[!0-9]*) [ -z "$x" ] || { echo "STOP: $v=$x: a port number"; exit 2; } ;; esac
  done
  # the line to run again after a fix, with the options given; every other name goes to quickstart, which refuses one it
  # does not know (a typo stops here, not silently)
  again='docker run --rm --network host -e USER="$USER"'
  a=
  for v in $(env | sed -n 's/^\([A-Za-z_][A-Za-z0-9_]*\)=.*/\1/p' | sort -u); do
    case " $UP_OWN " in *" $v "*) continue ;; esac
    eval "x=\${$v-}"
    again="$again -e $v=$(q "$x")"
    case " $UP_HERE " in *" $v "*) continue ;; esac
    a="$a --set $(q "$v=$x")"
  done
  keys=
  for k in /host/.ssh/id_ed25519 /host/.ssh/id_ecdsa /host/.ssh/id_rsa; do [ -f "$k" ] && keys="$keys -i $k"; done
  ssh_host="ssh -F none -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR $keys $USER@127.0.0.1"
  if ! err=$($ssh_host true 2>&1); then
    echo "STOP: this container cannot ssh to the host as $USER (127.0.0.1): $err"
    echo "FIX:  on the host: a key without a passphrase that opens this user's own account, sshd on port 22, --network host on the docker run line:"
    echo "      [ -f ~/.ssh/id_ed25519 ] || ssh-keygen -t ed25519 -N '' -f ~/.ssh/id_ed25519; cat ~/.ssh/id_ed25519.pub >> ~/.ssh/authorized_keys; chmod 600 ~/.ssh/authorized_keys"
    echo "THEN: $again -v \"\$HOME\":/host IMAGE up"
    exit 2
  fi
  # the name this container was started with, as the nodes will start the ranks: docker on the host knows it
  image=${IMAGE:-}
  if [ -z "$image" ]; then
    cid=$(sed -n 's|.*/containers/\([0-9a-f]\{64\}\)/.*|\1|p' /proc/self/mountinfo | head -1)
    [ -z "$cid" ] || image=$($ssh_host "docker inspect -f '{{.Config.Image}}' $cid" 2>/dev/null) || image=
  fi
  [ -n "$image" ] || { echo "STOP: this container's image name could not be read from the host's docker"; echo "FIX:  add -e IMAGE=NAME (the name every node can pull or holds)"; exit 2; }
  again="$again -v \"\$HOME\":/host $image up"
  ver=$(head -1 /opt/glm53/VERSION)
  echo "[up] $image, release $ver"
  to=/host/$dir
  conf=$to/glm53.conf
  if [ ! -f "$to/SHA256SUMS" ] || ! cmp -s /opt/glm53/SHA256SUMS "$to/SHA256SUMS"; then
    if [ -f "$to/tf-glm53" ] && $ssh_host "cd ~/$dir && ./tf-glm53 launcher status" 2>/dev/null | grep -q 'launcher: running'; then
      echo "STOP: ~/$dir holds another build and its launcher runs."
      echo "FIX:  stop the engine if it runs (AGENT-API.md section 7: saves the usage first), then: cd ~/$dir && ./tf-glm53 launcher stop"
      echo "THEN: $again"
      exit 2
    fi
    # banks.json is settings (where the banks are): an update keeps it
    keep=
    if [ -f "$to/expert-plan/banks.json" ]; then keep=$(mktemp); cp -p "$to/expert-plan/banks.json" "$keep"; fi
    mkdir -p "$to"
    cp -a /opt/glm53/. "$to"/
    [ -z "$keep" ] || { cp -p "$keep" "$to/expert-plan/banks.json"; rm -f "$keep"; }
    echo "[up] the release's files: ~/$dir"
  else
    echo "[up] the release's files: ~/$dir (already this build)"
  fi
  if [ ! -f "$conf" ]; then
    sed -e 's|^#RUNTIME=docker$|RUNTIME=docker|' -e "s|^#IMAGE=.*|IMAGE=$image|" "$to/glm53.t4.conf.example" > "$conf"
    echo "[up] settings: ~/$dir/glm53.conf (RUNTIME=docker, IMAGE=$image)"
  elif grep -q '^RUNTIME=docker' "$conf"; then
    sed -i "s|^IMAGE=.*|IMAGE=$image|" "$conf"
    echo "[up] settings: ~/$dir/glm53.conf kept (IMAGE=$image)"
  else
    echo "[up] settings: ~/$dir/glm53.conf kept (its RUNTIME is not docker: the ranks run as it says)"
  fi
  # the ports: before the launcher starts, which listens on them
  [ -z "${LAUNCHER_PORT:-}" ] && [ -n "${PORT:-}" ] && LAUNCHER_PORT=$((PORT + 10))
  for v in PORT LAUNCHER_PORT; do
    eval "x=\${$v:-}"
    [ -n "$x" ] && ! grep -q "^$v=$x\$" "$conf" || continue
    if $ssh_host "cd ~/$dir && ./tf-glm53 launcher status" 2>/dev/null | grep -q 'launcher: running'; then
      echo "STOP: $v=$x: this lane's launcher runs on the port of ~/$dir/glm53.conf"
      echo "FIX:  cd ~/$dir && ./tf-glm53 launcher stop (stop the engine first if it runs: AGENT-API.md section 7)"
      echo "THEN: $again"
      exit 2
    fi
    if grep -q "^$v=" "$conf"; then sed -i "s|^$v=.*|$v=$x|" "$conf"; else echo "$v=$x" >> "$conf"; fi
    echo "[up] settings: $v=$x"
  done
  chown -R "$(stat -c %u:%g /host)" "$to"
  a="--again $(q "$again") --image $(q "$image")$a"
  rc=0
  $ssh_host "cd ~/$dir && command -v python3 >/dev/null || { echo 'STOP: no python3 on the host (the launcher and the log keeper need it)'; echo 'FIX:  sudo apt-get install -y python3'; exit 2; }; ./tf-glm53 quickstart $a" || rc=$?
  exit $rc
}
case "${1:-help}" in
  up) cmd_up ;;
  install)
    [ -n "$2" ] && [ -d "$2" ] || { echo "install DIR: a directory mounted from the host (-v \"\$PWD\":/out ... install /out)"; exit 2; }
    # a directory Docker made for the mount is root's: the user (--user) cannot write it
    [ -w "$2" ] || { echo "install: $2 is not writable as uid $(id -u) (make the directory before docker run: mkdir glm53)"; exit 2; }
    # as root (no --user): the files the directory's owner can change (read first: cp -a gives the directory /opt/glm53's owner)
    owner=$(stat -c %u:%g "$2")
    cp -a /opt/glm53/. "$2"/
    [ "$(id -u)" != 0 ] || chown -R "$owner" "$2"
    echo "installed $(head -1 /opt/glm53/VERSION) in $2"
    echo "next: the whole way to a running engine in one line, on the first node: $UP_LINE"
    echo "      or by hand: cp glm53.t4.conf.example glm53.conf, edit it, ./tf-glm53 up. For an AI: AI-GUIDE.md, AGENT-API.md" ;;
  serve)
    [ -n "$2" ] || { echo "serve LOG ARGS..."; exit 2; }
    log=$2; shift 2
    [ "$log" = - ] && exec /opt/glm53/bin/tensorfold-native serve "$@" 2>&1
    exec /opt/glm53/bin/tensorfold-native serve "$@" > "$log" 2>&1 ;;
  version) cat /opt/glm53/VERSION ;;
  *) sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//'; [ "$1" = help ] || exit 2 ;;
esac
