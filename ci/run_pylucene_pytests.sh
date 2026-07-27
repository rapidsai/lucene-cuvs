#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

MVN_BIN="${MVN:-mvn}"
PYTHON_BIN="${PYTHON:-python3}"
SKIP_BUILD=0
FULL_END_TO_END=0
PYLUCENE_CASES="${CUVS_LUCENE_PYLUCENE_CASES:-}"
PYLUCENE_ROWS="${CUVS_LUCENE_PYLUCENE_ROWS:-}"
PYLUCENE_DIMS="${CUVS_LUCENE_PYLUCENE_DIMS:-}"
PYLUCENE_TOPK="${CUVS_LUCENE_PYLUCENE_TOPK:-}"
PYLUCENE_MIN_RECALL="${CUVS_LUCENE_PYLUCENE_MIN_RECALL:-}"
CUVS_LUCENE_JAR_PATH="${CUVS_LUCENE_JAR:-}"
PYLUCENE_TEST_CLASSES_PATH="${CUVS_LUCENE_PYLUCENE_TEST_CLASSES:-target/test-classes}"
PYTEST_ARGS=()

while [[ "$#" -gt 0 ]]; do
  arg="$1"
  shift
  case "${arg}" in
    --full-e2e|--gpu-e2e)
      FULL_END_TO_END=1
      ;;
    --no-build)
      SKIP_BUILD=1
      ;;
    --cases=*)
      PYLUCENE_CASES="${arg#--cases=}"
      ;;
    --rows=*)
      PYLUCENE_ROWS="${arg#--rows=}"
      ;;
    --dims=*)
      PYLUCENE_DIMS="${arg#--dims=}"
      ;;
    --topk=*)
      PYLUCENE_TOPK="${arg#--topk=}"
      ;;
    --min-recall=*)
      PYLUCENE_MIN_RECALL="${arg#--min-recall=}"
      ;;
    --)
      PYTEST_ARGS+=("$@")
      break
      ;;
    -h|--help)
      echo "Usage: ./test_pylucene.sh [OPTIONS] [-- PYTEST_ARGS...]"
      echo
      echo "Prepares the PyLucene JVM, jar, classpath, and native-library inputs, then runs pytest."
      echo
      echo "Options:"
      echo "  --no-build             Use existing Maven artifacts."
      echo "  --full-e2e             Run the complete CPU/GPU end-to-end suite."
      echo "  --cases=CASE[,CASE...] Select case names or groups."
      echo "  --rows=N               Set the minimum document count per scenario."
      echo "  --dims=N               Set vector dimensions."
      echo "  --topk=N               Set requested neighbor count."
      echo "  --min-recall=FLOAT     Set the brute-force recall floor (default: 0.75)."
      echo "  -- PYTEST_ARGS         Forward remaining arguments to pytest."
      echo
      echo "Execution-path groups:"
      echo "  cpu-hnsw, gpu-cagra-built-hnsw, gpu-cagra-search, gpu"
      echo
      echo "Behavior groups:"
      echo "  execution-paths, segment-topologies, force-merges,"
      echo "  hnsw-layer-counts, cagra-search-widths,"
      echo "  documents-without-vectors, deleted-documents,"
      echo "  all-but-one-document-deleted, single-document-index,"
      echo "  document-filter, all"
      echo
      echo "Environment overrides:"
      echo "  CUVS_LUCENE_JAR, CUVS_LUCENE_CUVS_JAVA_JAR,"
      echo "  CUVS_LUCENE_PYLUCENE_TEST_CLASSES,"
      echo "  CUVS_LUCENE_PYLUCENE_CASES, CUVS_LUCENE_PYLUCENE_ROWS,"
      echo "  CUVS_LUCENE_PYLUCENE_DIMS, CUVS_LUCENE_PYLUCENE_TOPK,"
      echo "  CUVS_LUCENE_PYLUCENE_MIN_RECALL, PYTHON, MVN"
      exit 0
      ;;
    *)
      echo "Unknown argument: ${arg}" >&2
      echo "Use -- before arguments intended for pytest." >&2
      exit 2
      ;;
  esac
done

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Required command not found: $1" >&2
    exit 127
  fi
}

absolute_from_repo_root() {
  case "$1" in
    /*)
      printf '%s\n' "$1"
      ;;
    *)
      printf '%s/%s\n' "${REPO_ROOT}" "$1"
      ;;
  esac
}

find_cuvs_java_jar() {
  if [[ -n "${CUVS_LUCENE_CUVS_JAVA_JAR:-}" ]]; then
    printf '%s\n' "${CUVS_LUCENE_CUVS_JAVA_JAR}"
    return
  fi

  local versioned_jar="${HOME}/.m2/repository/com/nvidia/cuvs/cuvs-java/${project_version}/cuvs-java-${project_version}.jar"
  if [[ -f "${versioned_jar}" ]]; then
    printf '%s\n' "${versioned_jar}"
    return
  fi

  local m2_repo="${HOME}/.m2/repository/com/nvidia/cuvs/cuvs-java"
  if [[ ! -d "${m2_repo}" ]]; then
    return
  fi

  find "${m2_repo}" \
    -type f \
    -name 'cuvs-java-*.jar' \
    ! -name '*sources*' \
    ! -name '*javadoc*' \
    ! -name '*x86_64*' \
    | sort -V \
    | tail -n 1
}

require_command "${PYTHON_BIN}"

if [[ "${SKIP_BUILD}" -eq 0 && -z "${CUVS_LUCENE_JAR_PATH}" ]]; then
  require_command "${MVN_BIN}"
  "${MVN_BIN}" clean package -DskipTests
fi

project_version="$(
  sed -n 's/.*CUVS_LUCENE#VERSION_UPDATE_MARKER_START--><version>\([^<]*\)<\/version>.*/\1/p' pom.xml \
    | head -n 1
)"
if [[ -z "${project_version}" ]]; then
  echo "Unable to determine project version from pom.xml" >&2
  exit 1
fi

if [[ -n "${CUVS_LUCENE_JAR_PATH}" ]]; then
  cuvs_lucene_jar="${CUVS_LUCENE_JAR_PATH}"
else
  cuvs_lucene_jar="target/cuvs-lucene-${project_version}.jar"
fi
cuvs_lucene_jar_abs="$(absolute_from_repo_root "${cuvs_lucene_jar}")"
if [[ ! -f "${cuvs_lucene_jar_abs}" ]]; then
  echo "cuvs-lucene jar not found: ${cuvs_lucene_jar_abs}" >&2
  echo "Run without --no-build, or set CUVS_LUCENE_JAR." >&2
  exit 1
fi

cuvs_java_jar="$(find_cuvs_java_jar)"
if [[ -z "${cuvs_java_jar}" || ! -f "${cuvs_java_jar}" ]]; then
  echo "Base cuvs-java jar not found." >&2
  echo "Set CUVS_LUCENE_CUVS_JAVA_JAR to the base jar, not a native classifier jar." >&2
  exit 1
fi
cuvs_java_jar_abs="$(absolute_from_repo_root "${cuvs_java_jar}")"

pylucene_test_classes_abs="$(
  absolute_from_repo_root "${PYLUCENE_TEST_CLASSES_PATH}"
)"
if [[ ! -d "${pylucene_test_classes_abs}" ]]; then
  echo "PyLucene test classes not found: ${pylucene_test_classes_abs}" >&2
  echo "Run Maven test compilation, or set CUVS_LUCENE_PYLUCENE_TEST_CLASSES." >&2
  exit 1
fi
pylucene_test_classes_abs="$(
  cd "${pylucene_test_classes_abs}" && pwd -P
)"

"${PYTHON_BIN}" -c "import lucene" >/dev/null 2>&1 || {
  echo "Python cannot import PyLucene's lucene module." >&2
  echo "Activate a PyLucene environment compatible with this project's Lucene version." >&2
  exit 1
}

"${PYTHON_BIN}" -m pytest --version >/dev/null 2>&1 || {
  echo "Python cannot run pytest. Install pytest in the active PyLucene environment." >&2
  exit 1
}

pytest_env=(
  "CUVS_LUCENE_JAR=${cuvs_lucene_jar_abs}"
  "CUVS_LUCENE_CUVS_JAVA_JAR=${cuvs_java_jar_abs}"
  "CUVS_LUCENE_PYLUCENE_TEST_CLASSES=${pylucene_test_classes_abs}"
  "CUVS_LUCENE_VERIFY_ALL_CODECS=${CUVS_LUCENE_VERIFY_ALL_CODECS:-1}"
)

if [[ "${FULL_END_TO_END}" -eq 1 ]]; then
  pytest_env+=(
    "CUVS_LUCENE_PYLUCENE_CASES=${PYLUCENE_CASES:-all}"
    "CUVS_LUCENE_PYLUCENE_ROWS=${PYLUCENE_ROWS:-2000}"
    "CUVS_LUCENE_PYLUCENE_DIMS=${PYLUCENE_DIMS:-32}"
    "CUVS_LUCENE_PYLUCENE_TOPK=${PYLUCENE_TOPK:-20}"
  )
else
  if [[ -n "${PYLUCENE_CASES}" ]]; then
    pytest_env+=("CUVS_LUCENE_PYLUCENE_CASES=${PYLUCENE_CASES}")
  fi
  if [[ -n "${PYLUCENE_ROWS}" ]]; then
    pytest_env+=("CUVS_LUCENE_PYLUCENE_ROWS=${PYLUCENE_ROWS}")
  fi
  if [[ -n "${PYLUCENE_DIMS}" ]]; then
    pytest_env+=("CUVS_LUCENE_PYLUCENE_DIMS=${PYLUCENE_DIMS}")
  fi
  if [[ -n "${PYLUCENE_TOPK}" ]]; then
    pytest_env+=("CUVS_LUCENE_PYLUCENE_TOPK=${PYLUCENE_TOPK}")
  fi
fi

if [[ -n "${PYLUCENE_MIN_RECALL}" ]]; then
  pytest_env+=("CUVS_LUCENE_PYLUCENE_MIN_RECALL=${PYLUCENE_MIN_RECALL}")
fi

env "${pytest_env[@]}" \
  "${PYTHON_BIN}" -m pytest -q -s \
  src/test/python/test_pylucene_end_to_end.py \
  "${PYTEST_ARGS[@]}"
