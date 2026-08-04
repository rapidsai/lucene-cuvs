/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import java.io.IOException;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.Map;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.hnsw.HnswGraphProvider;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.FilterLeafReader;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.QueryTimeout;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.knn.KnnCollectorManager;
import org.apache.lucene.util.Bits;

/**
 * Test-only adapters that PyLucene creates reflectively. Public no-argument constructors read query
 * configuration from Java system properties.
 */
public final class PyLuceneTestSupport {

  public static final String QUERY_FIELD_PROPERTY = "cuvs.lucene.pylucene.query.field";
  public static final String QUERY_TARGET_PROPERTY = "cuvs.lucene.pylucene.query.target";
  public static final String QUERY_K_PROPERTY = "cuvs.lucene.pylucene.query.k";
  public static final String QUERY_I_TOP_K_PROPERTY = "cuvs.lucene.pylucene.query.iTopK";
  public static final String QUERY_SEARCH_WIDTH_PROPERTY = "cuvs.lucene.pylucene.query.searchWidth";
  public static final String QUERY_EXPECTED_HNSW_M_PROPERTY =
      "cuvs.lucene.pylucene.query.expectedHnswM";
  public static final String QUERY_FILTER_FIELD_PROPERTY = "cuvs.lucene.pylucene.query.filterField";
  public static final String QUERY_FILTER_VALUE_PROPERTY = "cuvs.lucene.pylucene.query.filterValue";

  private static final int CAGRA_GRAPH_DEGREE = 32;
  private static final int CAGRA_INTERMEDIATE_GRAPH_DEGREE = 64;

  private PyLuceneTestSupport() {}

  /** CAGRA codec with graph degree 32 and intermediate graph degree 64. */
  public static final class CagraSearchCodec extends CuVS2510GPUSearchCodec {

    public CagraSearchCodec() throws Exception {
      super(
          new GPUSearchParams.Builder()
              .withStrategy(GPUSearchParams.Strategy.CUSTOM)
              .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.NN_DESCENT)
              .withGraphDegree(CAGRA_GRAPH_DEGREE)
              .withIntermediateGraphDegree(CAGRA_INTERMEDIATE_GRAPH_DEGREE)
              .build());
    }
  }

  public static final class HnswGraphVerifyingQuery extends KnnFloatVectorQuery {

    private final int expectedM;

    public HnswGraphVerifyingQuery() {
      this(QueryProperties.fromSystemPropertiesForHnsw());
    }

    private HnswGraphVerifyingQuery(QueryProperties properties) {
      super(properties.field, properties.target, properties.k, properties.filter);
      expectedM = parsePositiveInt(QUERY_EXPECTED_HNSW_M_PROPERTY);
    }

    @Override
    protected TopDocs approximateSearch(
        LeafReaderContext context,
        Bits acceptDocs,
        int visitedLimit,
        KnnCollectorManager knnCollectorManager)
        throws IOException {
      requireExpectedHnswGraph(context);
      return super.approximateSearch(context, acceptDocs, visitedLimit, knnCollectorManager);
    }

    @Override
    protected TopDocs exactSearch(
        LeafReaderContext context, DocIdSetIterator acceptIterator, QueryTimeout queryTimeout)
        throws IOException {
      requireExpectedHnswGraph(context);
      return super.exactSearch(context, acceptIterator, queryTimeout);
    }

    private void requireExpectedHnswGraph(LeafReaderContext context) {
      KnnVectorsReader vectorsReader = vectorReaderForField(context, getField());
      if (!(vectorsReader instanceof HnswGraphProvider)) {
        throw new AssertionError(
            "HNSW graph-verifying query requires HnswGraphProvider for field '"
                + getField()
                + "', got "
                + className(vectorsReader));
      }

      var fieldInfo = context.reader().getFieldInfos().fieldInfo(getField());
      if (fieldInfo == null) {
        throw new AssertionError("HNSW field metadata has no field info for: " + getField());
      }

      int actualM = persistedHnswM(vectorsReader, getField(), fieldInfo.number);
      if (actualM != expectedM) {
        throw new AssertionError(
            "HNSW graph-verifying query expected persisted M "
                + expectedM
                + " for field '"
                + getField()
                + "', got "
                + actualM);
      }
    }

    private static int persistedHnswM(
        KnnVectorsReader vectorsReader, String fieldName, int fieldNumber) {
      // The PyLucene bindings do not expose persisted M through their wrapped HNSW graph API.
      try {
        Field fieldsField = vectorsReader.getClass().getDeclaredField("fields");
        fieldsField.setAccessible(true);
        Object fieldsValue = fieldsField.get(vectorsReader);
        Object fieldEntry;
        if (fieldsValue instanceof Map<?, ?> fields) {
          fieldEntry = fields.get(fieldName);
        } else {
          Method getAccessor = fieldsValue.getClass().getMethod("get", int.class);
          fieldEntry = getAccessor.invoke(fieldsValue, fieldNumber);
        }
        if (fieldEntry == null) {
          throw new AssertionError("Persisted HNSW metadata has no entry for field: " + fieldName);
        }
        Method mAccessor = fieldEntry.getClass().getDeclaredMethod("M");
        mAccessor.setAccessible(true);
        Object value = mAccessor.invoke(fieldEntry);
        if (!(value instanceof Integer persistedM)) {
          throw new AssertionError("Unexpected persisted HNSW M value: " + value);
        }
        return persistedM;
      } catch (ReflectiveOperationException | RuntimeException exception) {
        throw new AssertionError(
            "Unable to inspect persisted HNSW M for field '" + fieldName + "'", exception);
      }
    }
  }

  public static final class CagraSearchQuery extends GPUKnnFloatVectorQuery {

    public CagraSearchQuery() {
      this(QueryProperties.fromSystemProperties());
    }

    private CagraSearchQuery(QueryProperties properties) {
      super(
          properties.field,
          properties.target,
          properties.k,
          properties.filter,
          properties.iTopK,
          properties.searchWidth);
    }

    @Override
    protected TopDocs approximateSearch(
        LeafReaderContext context,
        Bits acceptDocs,
        int visitedLimit,
        KnnCollectorManager knnCollectorManager)
        throws IOException {
      requireCagraOnlyIndex(context);
      return super.approximateSearch(context, acceptDocs, visitedLimit, knnCollectorManager);
    }

    @Override
    protected TopDocs exactSearch(
        LeafReaderContext context, DocIdSetIterator acceptIterator, QueryTimeout queryTimeout) {
      throw new AssertionError("CAGRA search query must not use Lucene exact vector scoring");
    }

    private void requireCagraOnlyIndex(LeafReaderContext context) {
      KnnVectorsReader vectorsReader = vectorReaderForField(context, getField());
      if (!(vectorsReader instanceof CuVS2510GPUVectorsReader cuvsReader)) {
        throw new AssertionError(
            "CAGRA search query requires CuVS2510GPUVectorsReader for field '"
                + getField()
                + "', got "
                + className(vectorsReader));
      }

      var fieldInfo = cuvsReader.getFieldInfos().fieldInfo(getField());
      if (fieldInfo == null) {
        throw new AssertionError(
            "CAGRA search field is absent from the cuVS reader: " + getField());
      }
      var cuvsIndexes = cuvsReader.getCuvsIndexes();
      if (cuvsIndexes == null) {
        throw new AssertionError(
            "CAGRA search found no loaded GPU indexes for field: " + getField());
      }
      GPUIndex gpuIndex = cuvsIndexes.get(fieldInfo.number);
      if (gpuIndex == null) {
        throw new AssertionError("CAGRA search found no GPU index for field: " + getField());
      }
      var cagraIndex = gpuIndex.getCagraIndex();
      if (cagraIndex == null) {
        throw new AssertionError(
            "CAGRA search requires a CAGRA index for field '"
                + getField()
                + "'; a brute-force or CPU fallback is not allowed");
      }
      if (gpuIndex.getBruteforceIndex() != null) {
        throw new AssertionError(
            "CAGRA search requires a CAGRA-only index for field '"
                + getField()
                + "'; a brute-force index was also loaded");
      }

      long actualGraphDegree = cagraIndex.getGraph().columns();
      if (actualGraphDegree != CAGRA_GRAPH_DEGREE) {
        throw new AssertionError(
            "CAGRA search expected actual graph degree "
                + CAGRA_GRAPH_DEGREE
                + " for field '"
                + getField()
                + "', got "
                + actualGraphDegree);
      }
    }
  }

  private static KnnVectorsReader vectorReaderForField(
      LeafReaderContext context, String fieldName) {
    LeafReader leafReader = FilterLeafReader.unwrap(context.reader());
    if (!(leafReader instanceof CodecReader codecReader)) {
      throw new AssertionError(
          "Graph-verifying query requires a CodecReader leaf, got "
              + leafReader.getClass().getName());
    }

    KnnVectorsReader vectorsReader = codecReader.getVectorReader();
    if (vectorsReader instanceof PerFieldKnnVectorsFormat.FieldsReader fieldsReader) {
      vectorsReader = fieldsReader.getFieldReader(fieldName);
    }
    return vectorsReader;
  }

  private static String className(Object value) {
    return value == null ? "<null>" : value.getClass().getName();
  }

  private static final class QueryProperties {

    private final String field;
    private final float[] target;
    private final int k;
    private final int iTopK;
    private final int searchWidth;
    private final Query filter;

    private QueryProperties(
        String field, float[] target, int k, int iTopK, int searchWidth, Query filter) {
      this.field = field;
      this.target = target;
      this.k = k;
      this.iTopK = iTopK;
      this.searchWidth = searchWidth;
      this.filter = filter;
    }

    private static QueryProperties fromSystemProperties() {
      String field = requiredProperty(QUERY_FIELD_PROPERTY);
      float[] target = parseTarget(requiredProperty(QUERY_TARGET_PROPERTY));
      int k = parsePositiveInt(QUERY_K_PROPERTY);
      int iTopK = parsePositiveInt(QUERY_I_TOP_K_PROPERTY);
      int searchWidth = parsePositiveInt(QUERY_SEARCH_WIDTH_PROPERTY);
      Query filter = optionalFilterFromSystemProperties();
      if (iTopK < k) {
        throw new IllegalArgumentException(
            "Java system property "
                + QUERY_I_TOP_K_PROPERTY
                + " must be greater than or equal to "
                + QUERY_K_PROPERTY
                + " ("
                + k
                + "), got: "
                + iTopK);
      }
      return new QueryProperties(field, target, k, iTopK, searchWidth, filter);
    }

    private static QueryProperties fromSystemPropertiesForHnsw() {
      String field = requiredProperty(QUERY_FIELD_PROPERTY);
      float[] target = parseTarget(requiredProperty(QUERY_TARGET_PROPERTY));
      int k = parsePositiveInt(QUERY_K_PROPERTY);
      Query filter = optionalFilterFromSystemProperties();
      return new QueryProperties(field, target, k, k, 1, filter);
    }
  }

  private static Query optionalFilterFromSystemProperties() {
    String field = System.getProperty(QUERY_FILTER_FIELD_PROPERTY);
    String value = System.getProperty(QUERY_FILTER_VALUE_PROPERTY);
    if (field == null && value == null) {
      return null;
    }
    if (field == null || value == null) {
      throw new IllegalArgumentException(
          "Java system properties "
              + QUERY_FILTER_FIELD_PROPERTY
              + " and "
              + QUERY_FILTER_VALUE_PROPERTY
              + " must be set together");
    }
    return new TermQuery(
        new Term(
            requiredProperty(QUERY_FILTER_FIELD_PROPERTY),
            requiredProperty(QUERY_FILTER_VALUE_PROPERTY)));
  }

  private static String requiredProperty(String propertyName) {
    String value = System.getProperty(propertyName);
    if (value == null || value.trim().isEmpty()) {
      throw new IllegalArgumentException("Missing required Java system property: " + propertyName);
    }
    return value.trim();
  }

  private static int parsePositiveInt(String propertyName) {
    String value = requiredProperty(propertyName);
    final int parsed;
    try {
      parsed = Integer.parseInt(value);
    } catch (NumberFormatException exception) {
      throw new IllegalArgumentException(
          "Java system property "
              + propertyName
              + " must be a positive integer, got: '"
              + value
              + "'",
          exception);
    }
    if (parsed <= 0) {
      throw new IllegalArgumentException(
          "Java system property " + propertyName + " must be a positive integer, got: " + parsed);
    }
    return parsed;
  }

  private static float[] parseTarget(String value) {
    String[] values = value.split(",", -1);
    float[] target = new float[values.length];
    for (int index = 0; index < values.length; index++) {
      String component = values[index].trim();
      if (component.isEmpty()) {
        throw malformedTarget(value, index, null);
      }
      try {
        target[index] = Float.parseFloat(component);
      } catch (NumberFormatException exception) {
        throw malformedTarget(value, index, exception);
      }
      if (!Float.isFinite(target[index])) {
        throw malformedTarget(value, index, null);
      }
    }
    return target;
  }

  private static IllegalArgumentException malformedTarget(
      String value, int componentIndex, NumberFormatException cause) {
    String message =
        "Java system property "
            + QUERY_TARGET_PROPERTY
            + " must be a comma-separated list of finite floats; invalid component "
            + componentIndex
            + " in: '"
            + value
            + "'";
    return cause == null
        ? new IllegalArgumentException(message)
        : new IllegalArgumentException(message, cause);
  }
}
