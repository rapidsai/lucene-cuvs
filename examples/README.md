# Examples

This maven project contains basic examples that showcase how `cuvs-lucene` can be used.

## Prerequisites

- [Docker](https://www.docker.com/)
- [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- A machine with an Nvidia GPU

## Steps

If you are currently in this directory (and to be in the `cuvs-lucene's` root directory) do:

```sh
cd ..
```

Then do:

```sh
docker run --rm --gpus all --pull=always --volume $PWD:$PWD --workdir $PWD -it rapidsai/ci-conda:26.10-cuda13.3.0-ubuntu24.04-py3.13
```

Inside the docker container (and in the `cuvs-lucene's` root directory) do:

```sh
./ci/build_java.sh && conda activate java && cd examples
```

To run Accelerated HNSW example do:

```sh
mvn clean install && java -Djava.util.logging.config.file=src/main/resources/logging.properties -cp target/examples-26.10.0-jar-with-merged-services.jar com.nvidia.cuvs.lucene.examples.AcceleratedHnswExample
```

To run the Index and Search on GPU example do:

```sh
mvn clean install && java -Djava.util.logging.config.file=src/main/resources/logging.properties -cp target/examples-26.10.0-jar-with-merged-services.jar com.nvidia.cuvs.lucene.examples.IndexAndSearchonGPUExample
```

To run the chunked `.fbin` ingestion example (reference pattern for streaming a large vector file
into an accelerated HNSW index without per-vector file reopening or holding the whole file in
memory) do:

```sh
mvn clean install && java -Djava.util.logging.config.file=src/main/resources/logging.properties -cp target/examples-26.08.0-jar-with-merged-services.jar com.nvidia.cuvs.lucene.examples.ChunkedFbinIngestExample
```

With no arguments it generates and indexes a small demo `.fbin`; pass a real file and chunk size as
`... ChunkedFbinIngestExample <path-to.fbin> <chunkSizeMB>`.
