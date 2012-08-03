package org.apache.mahout.cf.taste.hadoop.user;

import java.io.IOException;
import java.util.Iterator;

import org.apache.mahout.cf.taste.hadoop.item.PrefAndSimilarityColumnWritable;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.VarLongWritable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.DoubleFunction;

/**
 * Utility class that adds vectors together.
 */
public final class VectorAdditionUtils {

  private static final float BOOLEAN_PREF_VALUE = 1.0f;
  private static final DoubleFunction ABSOLUTE_VALUES = new DoubleFunction() {
    @Override
    public double apply(double value) {
      return value < 0 ? value * -1 : value;
    }
  };
  
  public VectorAdditionUtils() { }

  public static Vector reduceBooleanData(VarLongWritable userID,
      Iterable<PrefAndSimilarityColumnWritable> values)
      throws IOException, InterruptedException {
    /*
     * having boolean data, each estimated preference can only be 1, however we
     * can't use this to rank the recommended items, so we use the sum of
     * similarities for that.
     */
    Vector predictionVector = null;
    for (PrefAndSimilarityColumnWritable prefAndSimilarityColumn : values) {
      predictionVector = predictionVector == null ? prefAndSimilarityColumn
          .getSimilarityColumn() : predictionVector
          .plus(prefAndSimilarityColumn.getSimilarityColumn());
    }

    return predictionVector;
  }

  public static Vector reduceNonBooleanData(VarLongWritable userID,
      Iterable<PrefAndSimilarityColumnWritable> values)
      throws IOException, InterruptedException {
    /* each entry here is the sum in the numerator of the prediction formula */
    Vector numerators = null;
    /* each entry here is the sum in the denominator of the prediction formula */
    Vector denominators = null;
    /*
     * each entry here is the number of similar items used in the prediction
     * formula
     */
    Vector numberOfSimilarItemsUsed = new RandomAccessSparseVector(
        Integer.MAX_VALUE, 100);

    for (PrefAndSimilarityColumnWritable prefAndSimilarityColumn : values) {
      Vector simColumn = prefAndSimilarityColumn.getSimilarityColumn();
      float prefValue = prefAndSimilarityColumn.getPrefValue();
      /* count the number of items used for each prediction */
      Iterator<Vector.Element> usedItemsIterator = simColumn.iterateNonZero();
      while (usedItemsIterator.hasNext()) {
        int itemIDIndex = usedItemsIterator.next().index();
        numberOfSimilarItemsUsed.setQuick(itemIDIndex, numberOfSimilarItemsUsed
            .getQuick(itemIDIndex) + 1);
      }

      numerators = numerators == null ? prefValue == BOOLEAN_PREF_VALUE ? simColumn
          .clone()
          : simColumn.times(prefValue)
          : numerators.plus(prefValue == BOOLEAN_PREF_VALUE ? simColumn
              : simColumn.times(prefValue));

      simColumn.assign(ABSOLUTE_VALUES);
      denominators = denominators == null ? simColumn : denominators
          .plus(simColumn);
    }

    if (numerators == null) {
      return null;
    }

    Vector recommendationVector = new RandomAccessSparseVector(
        Integer.MAX_VALUE, 100);
    Iterator<Vector.Element> iterator = numerators.iterateNonZero();
    while (iterator.hasNext()) {
      Vector.Element element = iterator.next();
      int itemIDIndex = element.index();
      /* preference estimations must be based on at least 2 datapoints */
      if (numberOfSimilarItemsUsed.getQuick(itemIDIndex) > 1) {
        /* compute normalized prediction */
        double prediction = element.get() / denominators.getQuick(itemIDIndex);
        recommendationVector.setQuick(itemIDIndex, prediction);
      }
    }

    return recommendationVector;
  }

}
