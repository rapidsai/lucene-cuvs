/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

/** Formats vector-writer configuration for diagnostic output. */
final class WriterTelemetry {

  private WriterTelemetry() {}

  static String forCagra(GPUSearchParams params) {
    return "writerPath=gpu-cagra"
        + ";cagraStrategy="
        + params.getStrategy().name()
        + ";cagraGraphBuildAlgo="
        + params.getCagraGraphBuildAlgo().name()
        + ";cagraGraphDegree="
        + params.getGraphdegree()
        + ";cagraIntermediateGraphDegree="
        + params.getIntermediateGraphDegree();
  }

  static String forHnsw(AcceleratedHNSWParams params) {
    String writerPath;
    if (ThreadLocalCuVSResourcesProvider.isSupported()) {
      writerPath = "gpu-hnsw";
    } else if (ThreadLocalCuVSResourcesProvider.isCpuHnswFallbackForced()) {
      writerPath = "cpu-hnsw-fallback";
    } else {
      writerPath = "cpu-hnsw-auto-fallback";
    }

    return "writerPath="
        + writerPath
        + ";hnswLayers="
        + params.getHnswLayers()
        + ";cagraGraphBuildAlgo="
        + params.getCagraGraphBuildAlgo().name()
        + ";cagraGraphDegree="
        + params.getGraphdegree()
        + ";cagraIntermediateGraphDegree="
        + params.getIntermediateGraphDegree();
  }
}
