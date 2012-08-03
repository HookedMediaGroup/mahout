package org.apache.mahout.cf.taste.hadoop.user;

import java.io.IOException;
import java.util.Comparator;
import java.util.Iterator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.cf.taste.common.TopK;
import org.apache.mahout.cf.taste.hadoop.TasteHadoopUtils;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.recommender.GenericRecommendedItem;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.iterator.FileLineIterable;
import org.apache.mahout.math.VarLongWritable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.map.OpenIntLongHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.primitives.Floats;

/**
 * Generates recommendations for a user based on a vector of items. If an item
 * map is provided, then it is applied to decode the item ids.
 */
public final class RecommendationsWriter {
  
  private static final Logger log = LoggerFactory
      .getLogger(RecommendationsWriter.class);
  
  public static final String ITEMID_INDEX_PATH = "itemIDIndexPath";
  public static final String NUM_RECOMMENDATIONS = "numRecommendations";
  public static final int DEFAULT_NUM_RECOMMENDATIONS = 10;
  public static final String ITEMS_FILE = "itemsFile";
  
  private int recommendationsPerUser;
  private FastIDSet itemsToRecommendFor;
  private OpenIntLongHashMap indexItemIDMap;
  
  public static final Comparator<RecommendedItem> BY_PREFERENCE_VALUE = new Comparator<RecommendedItem>() {
    @Override
    public int compare(RecommendedItem one, RecommendedItem two) {
      return Floats.compare(one.getValue(), two.getValue());
    }
  };
  
  public RecommendationsWriter(Configuration conf) throws IOException {
    recommendationsPerUser = conf.getInt(NUM_RECOMMENDATIONS,
        DEFAULT_NUM_RECOMMENDATIONS);
    
    String itemIdIndexFilePathString = conf.get(ITEMID_INDEX_PATH, "");
    if (!itemIdIndexFilePathString.equals("")) {
      indexItemIDMap = TasteHadoopUtils.readItemIDIndexMap(
          itemIdIndexFilePathString, conf);
    } else {
      indexItemIDMap = null;
    }
    
    String itemFilePathString = conf.get(ITEMS_FILE);
    if (itemFilePathString != null) {
      itemsToRecommendFor = new FastIDSet();
      for (String line : new FileLineIterable(HadoopUtil.openStream(new Path(
          itemFilePathString), conf))) {
        try {
          itemsToRecommendFor.add(Long.parseLong(line));
        } catch (NumberFormatException nfe) {
          log.warn("itemsFile line ignored: {}", line);
        }
      }
    } else {
      itemsToRecommendFor = null;
    }
  }
  
  /**
   * find the top entries in recommendationVector, map them to the real itemIDs
   * and write back the result
   */
  public TopK<RecommendedItem> writeRecommendedItems(VarLongWritable userID,
      Vector recommendationVector) throws IOException, InterruptedException {
    
    TopK<RecommendedItem> topKItems = new TopK<RecommendedItem>(
        recommendationsPerUser, RecommendationsWriter.BY_PREFERENCE_VALUE);
    
    Iterator<Vector.Element> recommendationVectorIterator = recommendationVector
        .iterateNonZero();
    while (recommendationVectorIterator.hasNext()) {
      Vector.Element element = recommendationVectorIterator.next();
      int index = element.index();
      long itemID;
      if (indexItemIDMap != null && !indexItemIDMap.isEmpty()) {
        itemID = indexItemIDMap.get(index);
      } else { // we don't have any mappings, so just use the original
        itemID = index;
      }
      if (itemsToRecommendFor == null || itemsToRecommendFor.contains(itemID)) {
        float value = (float) element.get();
        if (!Float.isNaN(value)) {
          topKItems.offer(new GenericRecommendedItem(itemID, value));
        }
      }
    }
    
    return topKItems;
  }
  
}
