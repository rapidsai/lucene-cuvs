# Lucene cuVS Integration

This is a codec for connecting [cuVS](https://github.com/rapidsai/cuvs), NVIDIA's GPU accelerated vector search library, into [Apache Lucene](https://github.com/apache/lucene).

## Overview

The cuVS library is plugged in as a new KnnVectorFormat via a custom codec.

> [!CAUTION]
> This is not production ready yet.

## Building

Install NVIDIA drivers, CUDA 12.8, Maven 3.9.6+ and JDK 22.

```sh
mvn clean compile package
```

The artifacts would be built and available in the target/ folder.

> [!NOTE]
> The code style format is automatically enforced (including the missing license header, if any) using the Spotless maven plugin. This currently happens in the maven's `validate` stage.