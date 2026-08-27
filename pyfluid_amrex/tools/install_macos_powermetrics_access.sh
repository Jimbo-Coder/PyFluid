#!/bin/sh

# One-time privileged installer for the narrowly scoped PyFluid powermetrics
# wrapper. The installed wrapper is root-owned, accepts only a bounded duration,
# has no output-path option, and executes one fixed Apple tool invocation.

set -eu

installed_wrapper=/usr/local/sbin/pyfluid-powermetrics
sudoers_file=/private/etc/sudoers.d/pyfluid-powermetrics

if [ "$(/usr/bin/id -u)" -ne 0 ]; then
    /usr/bin/printf '%s\n' \
        "run this installer once with sudo: sudo $0" >&2
    exit 77
fi

target_user=${SUDO_USER:-}
if [ -z "$target_user" ] || [ "$target_user" = root ]; then
    /usr/bin/printf '%s\n' \
        "the installer must be invoked by the target user through sudo" >&2
    exit 77
fi
case "$target_user" in
    *[!A-Za-z0-9_.-]*)
        /usr/bin/printf '%s\n' "unsupported sudo user name" >&2
        exit 65
        ;;
esac

script_dir=$(CDPATH= cd -- "$(/usr/bin/dirname -- "$0")" && /bin/pwd)
source_wrapper=$script_dir/macos_powermetrics_capture.sh
if [ ! -f "$source_wrapper" ]; then
    /usr/bin/printf 'missing wrapper source: %s\n' "$source_wrapper" >&2
    exit 66
fi

if ! /usr/bin/grep -Eq \
    '^[[:space:]]*[#@]includedir[[:space:]]+(/private)?/etc/sudoers\.d([[:space:]]|$)' \
    /private/etc/sudoers; then
    /usr/bin/printf '%s\n' \
        "/private/etc/sudoers does not include /private/etc/sudoers.d; no changes made" >&2
    exit 78
fi

temporary_sudoers=$(/usr/bin/mktemp /private/tmp/pyfluid-powermetrics-sudoers.XXXXXX)
cleanup() {
    /bin/rm -f "$temporary_sudoers"
}
trap cleanup EXIT HUP INT TERM

/usr/bin/printf '%s ALL=(root) NOPASSWD: %s\n' \
    "$target_user" "$installed_wrapper" > "$temporary_sudoers"
/bin/chmod 0440 "$temporary_sudoers"
/usr/sbin/visudo -cf "$temporary_sudoers"

/bin/mkdir -p /usr/local/sbin /private/etc/sudoers.d
/usr/bin/install -o root -g wheel -m 0755 \
    "$source_wrapper" "$installed_wrapper"
/usr/bin/install -o root -g wheel -m 0440 \
    "$temporary_sudoers" "$sudoers_file"
if ! /usr/sbin/visudo -cf /private/etc/sudoers; then
    /bin/rm -f "$sudoers_file"
    /usr/bin/printf '%s\n' \
        "combined sudoers validation failed; the passwordless rule was removed" >&2
    exit 78
fi

/usr/bin/printf '%s\n' \
    "installed passwordless fixed-function capture for $target_user" \
    "test with:" \
    "  sudo -n $installed_wrapper 15 > /tmp/pyfluid-powermetrics-test.txt"
