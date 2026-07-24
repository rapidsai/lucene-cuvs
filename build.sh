#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e -u -o pipefail

ARGS="$*"
NUMARGS=$#

VERSION="26.08.0" # Note: The version is updated automatically when ci/release/update-version.sh is invoked
GROUP_ID="com.nvidia.cuvs.lucene"

function hasArg {
    (( NUMARGS != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

# Checks whether the checked out cuvs branch modifies the C/C++ sources of libcuvs
# with respect to the branch it is based on. Must be invoked from within the cuvs
# clone.
function cuvsModifiesCpp {
  local base_ref merge_base

  # The branch a pull request is based on cannot be derived from the pull request
  # ref itself, so fall back on the release branch matching the cuvs version and,
  # if that one does not exist yet, on the default branch of the repository.
  base_ref="${CUVS_BASE_REF:-}"
  if [[ -z "$base_ref" ]]; then
    base_ref="release/$(sed -E 's/^([0-9]+)\.([0-9]+).*$/\1.\2/' VERSION)"
    if ! git rev-parse --verify --quiet "origin/$base_ref" > /dev/null; then
      base_ref=$(git symbolic-ref --short refs/remotes/origin/HEAD)
      base_ref="${base_ref#origin/}"
    fi
  fi

  if ! merge_base=$(git merge-base "origin/$base_ref" HEAD 2> /dev/null); then
    echo "Could not determine the merge base with 'origin/$base_ref', assuming libcuvs was modified."
    return 0
  fi

  echo "Comparing against 'origin/$base_ref' to detect changes in the cuvs C/C++ sources."
  git diff --name-only "$merge_base" HEAD -- cpp/ \
    | grep -qE '(\.(c|cc|cpp|cxx|cu|cuh|h|hpp|hxx|cmake)|CMakeLists\.txt)$'
}

if hasArg --build-cuvs-java; then
  CUVS_WORKDIR="cuvs-workdir"
  CUVS_GIT_REPO="https://github.com/rapidsai/cuvs.git"
  BRANCH=$(cat "RAPIDS_BRANCH")
  if [[ -d "$CUVS_WORKDIR" && -n "$(ls -A "$CUVS_WORKDIR")" ]]; then
    echo "Directory '$CUVS_WORKDIR' exists and is not empty."
    pushd $CUVS_WORKDIR
    git pull
  else
    echo "Directory '$CUVS_WORKDIR' does not exist or is empty. Cloning the cuvs's '$BRANCH' branch."
    # Correct branch selection is crucial to avoid version mismatch issues when testing.
    git clone --branch "$BRANCH" $CUVS_GIT_REPO $CUVS_WORKDIR
    pushd $CUVS_WORKDIR
  fi

  # libcuvs comes from the conda packages, so normally only the java bindings have
  # to be built. A pull request that changes the C/C++ sources is not part of those
  # packages though, and the bindings would be built and tested against a stale
  # library, so CI asks for libcuvs to be built from source in that case.
  CUVS_BUILD_TARGETS=("java")
  if hasArg --build-libcuvs-if-changed && [[ "$BRANCH" == pull-request/* ]] && cuvsModifiesCpp; then
    echo "Branch '$BRANCH' modifies the cuvs C/C++ sources, building libcuvs from source as well."
    CUVS_BUILD_TARGETS=("libcuvs" "${CUVS_BUILD_TARGETS[@]}")
    # The libraries that were just built have to take precedence over the ones
    # provided by the conda packages, both here and while running the java tests.
    LD_LIBRARY_PATH="$PWD/cpp/build${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export LD_LIBRARY_PATH
    echo "LD_LIBRARY_PATH is: $LD_LIBRARY_PATH"
  fi
  ./build.sh "${CUVS_BUILD_TARGETS[@]}"
  popd
fi

MAVEN_VERIFY_ARGS=()
if ! hasArg --run-java-tests; then
  MAVEN_VERIFY_ARGS=("-DskipTests")
fi

mvn clean verify "${MAVEN_VERIFY_ARGS[@]}" \
  && mvn jacoco:report \
  && mvn install:install-file -Dfile=./target/cuvs-lucene-$VERSION.jar -DgroupId=$GROUP_ID -DartifactId=cuvs-lucene -Dversion=$VERSION -Dpackaging=jar \
  && cp pom.xml ./target/
