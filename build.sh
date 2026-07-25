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
  # to be built. For a pull request, the conda packages do not contain the PR's
  # changes, so CI downloads the pre-built libcuvs artifact from the PR's own CI run.
  CUVS_BUILD_TARGETS=("java")
  if hasArg --use-pr-libcuvs && [[ "$BRANCH" == pull-request/* ]]; then
    PR_NUM="${BRANCH#pull-request/}"
    CUDA_MAJOR="${RAPIDS_CUDA_VERSION%%.*}"
    LIBCUVS_ARTIFACT="cuvs_conda_cpp_libcuvs_$(arch)_cu${CUDA_MAJOR}"
    echo "Downloading libcuvs conda artifact '${LIBCUVS_ARTIFACT}' from cuvs PR #${PR_NUM}..."
    CUVS_COMMIT=$(gh pr view "$PR_NUM" --repo rapidsai/cuvs --json headRefOid --jq '.headRefOid')
    CUVS_RUN_ID=$(gh run list --repo rapidsai/cuvs --branch "pull-request/${PR_NUM}" --commit "$CUVS_COMMIT" \
      --workflow pr.yaml --json 'createdAt,databaseId' --jq 'sort_by(.createdAt) | reverse | .[0] | .databaseId')
    LIBCUVS_CONDA_DIR=$(mktemp -d)
    gh run download "$CUVS_RUN_ID" --repo rapidsai/cuvs --name "$LIBCUVS_ARTIFACT" --dir "$LIBCUVS_CONDA_DIR"
    LIBCUVS_DIR=$(rapids-extract-conda-files "$LIBCUVS_CONDA_DIR")
    # The downloaded library has to take precedence over the one provided by the
    # conda packages, both here and while running the java tests.
    LD_LIBRARY_PATH="$LIBCUVS_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
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
