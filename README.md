# Lucene cuVS

This is a project for using [cuVS](https://github.com/rapidsai/cuvs), NVIDIA's GPU accelerated vector search library, with [Apache Lucene](https://github.com/apache/lucene).

## Overview

This library provides a new [KnnVectorFormat](https://lucene.apache.org/core/10_3_1/core/org/apache/lucene/codecs/KnnVectorsFormat.html) which can be plugged into a Lucene codec.

## Building

### Prerequisites

- [CUDA 12.0+](https://developer.nvidia.com/cuda-toolkit-archive),
- [Maven 3.9.6+](https://maven.apache.org/download.cgi),
- [JDK 22](https://jdk.java.net/archive/)

```sh
mvn clean compile package
```

The artifacts would be built and available in the target / folder.

### Using with PyLucene

PyLucene embeds a JVM and starts it with the classpath passed to `lucene.initVM(...)`.
Because PyLucene's generated Python module only exposes the Java classes it was built
to wrap, use Lucene's service provider lookup to load `cuvs-lucene` codecs from
Python instead of importing `com.nvidia.cuvs.lucene` classes directly.

Build the standard cuvs-lucene jar:

```sh
mvn clean package -DskipTests
```

Then start PyLucene with the base `cuvs-java` jar, the standard `cuvs-lucene`
jar, and PyLucene's own Lucene classpath:

```python
import os
from pathlib import Path

import lucene

cuvs_java_jar = Path(os.environ["CUVS_LUCENE_CUVS_JAVA_JAR"])
cuvs_lucene_jar = next(
    jar
    for jar in Path("target").glob("cuvs-lucene-*.jar")
    if "-jar-with-" not in jar.name
    and not jar.name.endswith(("-sources.jar", "-javadoc.jar"))
)
lucene.initVM(
    classpath=os.pathsep.join(
        [str(cuvs_java_jar), str(cuvs_lucene_jar), lucene.CLASSPATH]
    ),
    vmargs=[
        "--enable-native-access=ALL-UNNAMED",
        "--add-modules=jdk.incubator.vector",
    ],
)

from org.apache.lucene.codecs import Codec

codec = Codec.forName("Lucene101AcceleratedHNSWCodec")
```

Use the returned `codec` with `IndexWriterConfig.setCodec(codec)`. The standard
artifact includes `cuvs-lucene` classes and service descriptors.
PyLucene must provide Lucene classes, and the base multi-release `cuvs-java` jar
must be present separately on the JVM classpath. Do not use a native classifier
`cuvs-java` jar here unless you also want to rely on its embedded native
libraries; the base jar uses native libraries from
`LD_LIBRARY_PATH`/`java.library.path`.

#### PyLucene end-to-end tests

Pytest cases are under `src/test/python`. The parametrized cases and assertions
are in `test_pylucene_end_to_end.py`; reusable index and search code is in
`pylucene_test_support.py`. The test-only Java adapters in
`src/test/java/com/nvidia/cuvs/lucene/PyLuceneTestSupport.java` are compiled to
`target/test-classes` and are not included in the published jar.

`ci/run_pylucene_pytests.sh` builds the Maven artifacts when requested, resolves
the PyLucene classpath inputs, and invokes pytest. `test_pylucene.sh` at the
repository root is the convenient entry point.

To build the artifacts and run the default CPU check in an activated PyLucene
environment:

```sh
./test_pylucene.sh
```

The default runs the jar-packaging checks and
`cpu-hnsw-single-document-index`. Use the groups below for broader coverage.
Use `--no-build` when the standard jar and test bridge are already compiled.
`CUVS_LUCENE_PYLUCENE_TEST_CLASSES` can point to a different test-classes
directory and defaults to `target/test-classes`.

Pytest can also be invoked directly once the PyLucene environment and classpath
inputs are available:

```sh
CUVS_LUCENE_JAR=/absolute/path/to/cuvs-lucene.jar \
CUVS_LUCENE_CUVS_JAVA_JAR=/absolute/path/to/cuvs-java.jar \
CUVS_LUCENE_PYLUCENE_TEST_CLASSES="$(pwd)/target/test-classes" \
python3 -m pytest -q -s src/test/python/test_pylucene_end_to_end.py
```

The execution-path groups are:

- `cpu-hnsw`: HNSW build and search through Lucene's CPU path.
- `gpu-cagra-built-hnsw`: GPU CAGRA build followed by HNSW search.
- `gpu-cagra-search`: GPU CAGRA build and search.

GPU cases assert that cuVS was actually used and fail if it is unavailable or
falls back to CPU. CPU cases assert and report the CPU path, and HNSW cases
verify the persisted graph shape. CAGRA cases use `graphDegree=32` and
`intermediateGraphDegree=64`.

Vectors and queries are deterministic. Expected neighbors are computed with
brute force. Queries for live indexed vectors check rank-one self matches,
duplicate hits, and a configurable recall floor. Separate tests cover segment
topology, force merges, HNSW layer count, CAGRA `searchWidth`, document filters,
live documents without vectors, deleted documents, and searches after all but
one document have been deleted. Set the recall floor with
`--min-recall=FLOAT` in the wrapper or
`CUVS_LUCENE_PYLUCENE_MIN_RECALL` for direct pytest execution.
The default floor is `0.75`.

Useful behavior groups include `execution-paths`, `segment-topologies`,
`force-merges`, `hnsw-layer-counts`, `cagra-search-widths`,
`documents-without-vectors`, `deleted-documents`,
`all-but-one-document-deleted`, `single-document-index`, and
`document-filter`.

Run the complete CPU/GPU end-to-end suite with:

```sh
./test_pylucene.sh --full-e2e
```

Select a focused group or set the minimum document count per scenario:

```sh
./test_pylucene.sh --cases=cagra-search-widths \
  --rows=5000 --dims=64 --topk=20 --min-recall=0.8
```

`--rows` is a lower bound. CAGRA cases use at least 97 vector-bearing
documents per constructed graph, one more than cuVS's internal NN-Descent
degree of 96. The three-layer HNSW case uses at least 24,832 vector-bearing
documents so its third layer retains 97. Arguments after `--` are passed to
pytest, for example:

```sh
./test_pylucene.sh --no-build --cases=gpu-cagra-search -- -x
```

### Running Tests

```sh
export LD_LIBRARY_PATH={ PATH TO YOUR LOCAL libcuvs_c.so }:$LD_LIBRARY_PATH && mvn clean test
```

## Contributing

> [!NOTE]
> The code style format is automatically enforced (including the missing license header, if any) using the [Spotless maven plugin](https://github.com/diffplug/spotless/tree/main/plugin-maven). This currently happens in the maven's `validate` stage.
