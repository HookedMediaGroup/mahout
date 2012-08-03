package org.apache.mahout.math.hadoop.similarity.cooccurrence;

import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.common.ClassUtils;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.similarity.cooccurrence.measures.VectorSimilarityMeasure;

public class ColumnVectorNormMapper extends Mapper<IntWritable,VectorWritable,IntWritable,VectorWritable> {

  private VectorSimilarityMeasure similarity;
  private Vector norms;
  private Vector nonZeroEntries;
  private Vector maxValues;
  private double threshold;

  @Override
  protected void setup(Context ctx) throws IOException, InterruptedException {
    similarity = ClassUtils.instantiateAs(ctx.getConfiguration().get(RowSimilarityJob.SIMILARITY_CLASSNAME),
        VectorSimilarityMeasure.class);
    norms = new RandomAccessSparseVector(Integer.MAX_VALUE);
    nonZeroEntries = new RandomAccessSparseVector(Integer.MAX_VALUE);
    maxValues = new RandomAccessSparseVector(Integer.MAX_VALUE);
    threshold = Double.parseDouble(ctx.getConfiguration().get(RowSimilarityJob.THRESHOLD));
  }

  @Override
  protected void map(IntWritable row, VectorWritable vectorWritable, Context ctx)
      throws IOException, InterruptedException {

    Vector rowVector = similarity.normalize(vectorWritable.get());

    int numNonZeroEntries = 0;
    double maxValue = Double.MIN_VALUE;

    Iterator<Vector.Element> nonZeroElements = rowVector.iterateNonZero();
    while (nonZeroElements.hasNext()) {
      Vector.Element element = nonZeroElements.next();
      RandomAccessSparseVector partialColumnVector = new RandomAccessSparseVector(Integer.MAX_VALUE);
      partialColumnVector.setQuick(row.get(), element.get());
      ctx.write(new IntWritable(element.index()), new VectorWritable(partialColumnVector));

      numNonZeroEntries++;
      if (maxValue < element.get()) {
        maxValue = element.get();
      }
    }

    if (threshold != RowSimilarityJob.NO_THRESHOLD) {
      nonZeroEntries.setQuick(row.get(), numNonZeroEntries);
      maxValues.setQuick(row.get(), maxValue);
    }
    norms.setQuick(row.get(), similarity.norm(rowVector));

    ctx.getCounter(RowSimilarityJob.Counters.ROWS).increment(1);
  }

  @Override
  protected void cleanup(Context ctx) throws IOException, InterruptedException {
    super.cleanup(ctx);
    // dirty trick
    ctx.write(new IntWritable(RowSimilarityJob.NORM_VECTOR_MARKER), new VectorWritable(norms));
    ctx.write(new IntWritable(RowSimilarityJob.NUM_NON_ZERO_ENTRIES_VECTOR_MARKER), new VectorWritable(nonZeroEntries));
    ctx.write(new IntWritable(RowSimilarityJob.MAXVALUE_VECTOR_MARKER), new VectorWritable(maxValues));
  }
}