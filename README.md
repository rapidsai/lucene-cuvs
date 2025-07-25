# Lucene cuVS

This is a codec for connecting [cuVS](https://github.com/rapidsai/cuvs), NVIDIA's GPU accelerated vector search library, into [Apache Lucene](https://github.com/apache/lucene).

## Overview

The cuVS library is plugged in as a new `KnnVectorFormat` via a custom codec.

> [!CAUTION]
> This is not production ready yet.

### Prerequisites
- [CUDA 12.8](https://developer.nvidia.com/cuda-12-8-0-download-archive),
- [Maven 3.9.6+](https://maven.apache.org/download.cgi),
- [JDK 22](https://jdk.java.net/archive/)

### Building
```sh
mvn clean compile package
```

The artifacts would be built and available in the target/ folder.

> [!NOTE]
> The code style format is automatically enforced (including the missing license header, if any) using the [Spotless maven plugin](https://github.com/diffplug/spotless/tree/main/plugin-maven). This currently happens in the maven's `validate` stage.