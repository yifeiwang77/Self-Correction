#!/usr/bin/env bash

USAGE="Run static code checkers (but not unit tests)

This script expects to be run from the repository's root directory so the tools
can find their configuration files.

USAGE:
    bash ${0} [--help]

OPTIONS:
    --help, -h: Show this help and exit

EXIT VALUES:
    - 0     Success
    - 1     Invalid option/argument
    - 2     Python syntax error
    - 4     Python format error (black, isort)
    - 8     Python linter error (mypy, pylint)
"

if [[ "${#}" = 1 ]]
then
    if [[ "${1}" = "-h" || "${1}" = "--help" ]]
    then
        echo "${USAGE}"
        exit 0
    fi
    echo "Invalid argument ${@}" >&2
    exit 1
elif [[ "${#}" > 1 ]]
then
    echo "Invalid argument ${@}" >&2
    exit 1
fi

REPO="$(dirname -- "${0}")"

black --check "${REPO}"
BLACK_STATUS="${?}"
if [[ "${BLACK_STATUS}" = 123 ]]
then
    # A black exit code of 123 usually indicates a synax error.
    # In that event, there's no reason to keep running the other checks.
    # They'll fail for the same reason.
    exit 2
fi

isort --check --diff "${REPO}"
ISORT_STATUS="${?}"

mypy "${REPO}"
MYPY_STATUS="${?}"

pylint "${REPO}"
PYLINT_STATUS="${?}"

EXIT_STATUS=0
if [[ "${BLACK_STATUS}" > 0 || "${ISORT_STATUS}" > 0 ]]
then
    EXIT_STATUS="$(( "${EXIT_STATUS}" + 4 ))"
fi
if [[ "${MYPY_STATUS}" > 0 || "${PYLINT_STATUS}" > 0 ]]
then
    EXIT_STATUS="$(( "${EXIT_STATUS}" + 8 ))"
fi
exit "${EXIT_STATUS}"
