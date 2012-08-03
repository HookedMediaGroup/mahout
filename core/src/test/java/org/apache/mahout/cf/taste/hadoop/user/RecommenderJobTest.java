package org.apache.mahout.cf.taste.hadoop.user;

import java.io.File;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.hadoop.conf.Configuration;
import org.apache.mahout.cf.taste.impl.TasteTestCase;
import org.apache.mahout.cf.taste.impl.recommender.GenericRecommendedItem;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.common.iterator.FileLineIterable;
import org.apache.mahout.math.hadoop.similarity.cooccurrence.measures.TanimotoCoefficientSimilarity;
import org.junit.Test;

import com.google.common.collect.Maps;

public class RecommenderJobTest extends TasteTestCase {

  /**
   * small integration test that runs the full job
   *
   * As a tribute to http://www.slideshare.net/srowen/collaborative-filtering-at-scale,
   * we recommend people food to animals in this test :)
   *
   * <pre>
   *
   *  user-item-matrix
   *
   *          burger  hotdog  berries  icecream
   *  dog       5       5        2        -
   *  rabbit    2       -        3        5
   *  cow       -       5        -        3
   *  donkey    3       -        -        5
   *
   *
   *  user-user-similarity-matrix (tanimoto-coefficient of the user-vectors of the user-item-matrix)
   *
   *           dog   rabbit  cow   donkey
   *  dog       -     0.5    0.25   0.25
   *  rabbit   0.5     -     0.25   0.66
   *  cow      0.25   0.25    -     0.33
   *  donkey   0.25   0.66   0.33    -
   *  
   *  
   *  Prediction(dog, icecream)   = (0.5 * 5 + 0.25 * 3 + 0.25 * 5) / (0.5 + 0.25 + 0.25)   ~ 4.5
   *  Prediction(rabbit, hotdog)  = (0.5 * 5 + 0.25 * 5) / (0.5 + 0.25)                     ~ 5.0
   *  Prediction(cow, burger)     = (0.25 * 5 + 0.25 * 2 + 0.33 * 3) / (0.25 + 0.25 + 0.33) ~ 3.3
   *  Prediction(cow, berries)    = (0.25 * 2 + 0.25 * 3) / (0.25 + 0.25)                   ~ 2.5
   *  Prediction(donkey, hotdog)  = (0.25 * 5 + 0.33 * 5) / (0.25 + 0.33)                   ~ 5.0
   *  Prediction(donkey, berries) = (0.25 * 2 + 0.66 * 3) / (0.25 + 0.66)                   ~ 2.7
   *
   * </pre>
   */
  @Test
  public void testCompleteJob() throws Exception {

    File inputFile = getTestTempFile("prefs.txt");
    File outputDir = getTestTempDir("output");
    outputDir.delete();
    File tmpDir = getTestTempDir("tmp");

    writeLines(inputFile,
        "1,1,5",
        "1,2,5",
        "1,3,2",
        "2,1,2",
        "2,3,3",
        "2,4,5",
        "3,2,5",
        "3,4,3",
        "4,1,3",
        "4,4,5");

    RecommenderJob recommenderJob = new RecommenderJob();

    Configuration conf = new Configuration();
    conf.set("mapred.input.dir", inputFile.getAbsolutePath());
    conf.set("mapred.output.dir", outputDir.getAbsolutePath());
    conf.setBoolean("mapred.output.compress", false);

    recommenderJob.setConf(conf);

    recommenderJob.run(new String[] {"--tempDir", tmpDir.getAbsolutePath(),
        "--similarityClassname", TanimotoCoefficientSimilarity.class.getName(),
        "--numRecommendations", "4", "--encodeLongsAsInts",
        Boolean.FALSE.toString(), "--itemBased", Boolean.FALSE.toString() });

    Map<Long,List<RecommendedItem>> recommendations = readRecommendations(new File(outputDir, "part-r-00000"));

    assertEquals(4, recommendations.size());

    for (Entry<Long,List<RecommendedItem>> entry : recommendations.entrySet()) {
      long userID = entry.getKey();
      List<RecommendedItem> items = entry.getValue();
      assertNotNull(items);
      RecommendedItem item1 = items.get(0);

      if (userID == 1L) {
        assertEquals(1, items.size());
        assertEquals(4L, item1.getItemID());
        assertEquals(4.5, item1.getValue(), 0.25);
      }
      if (userID == 2L) {
        assertEquals(1, items.size());
        assertEquals(2L, item1.getItemID());
        assertEquals(5.0, item1.getValue(), 0.25);
      }
      if (userID == 3L) {
        assertEquals(2, items.size());
        assertEquals(1L, item1.getItemID());
        assertEquals(3.3, item1.getValue(), 0.25);
        RecommendedItem item2 = items.get(1);
        assertEquals(3L, item2.getItemID());
        assertEquals(2.5, item2.getValue(), 0.25);
      }
      if (userID == 4L) {
        assertEquals(2, items.size());
        assertEquals(2L, item1.getItemID());
        assertEquals(5.0, item1.getValue(), 0.25);
        RecommendedItem item2 = items.get(1);
        assertEquals(3L, item2.getItemID());
        assertEquals(2.7, item2.getValue(), 0.25);
      }
    }
  }

  static Map<Long,List<RecommendedItem>> readRecommendations(File file) throws IOException {
    Map<Long,List<RecommendedItem>> recommendations = Maps.newHashMap();
    Iterable<String> lineIterable = new FileLineIterable(file);
    for (String line : lineIterable) {

      String[] keyValue = line.split("\t");
      long userID = Long.parseLong(keyValue[0]);
      String[] tokens = keyValue[1].replaceAll("\\[", "")
          .replaceAll("\\]", "").split(",");

      List<RecommendedItem> items = new LinkedList<RecommendedItem>();
      for (String token : tokens) {
        String[] itemTokens = token.split(":");
        long itemID = Long.parseLong(itemTokens[0]);
        float value = Float.parseFloat(itemTokens[1]);
        items.add(new GenericRecommendedItem(itemID, value));
      }
      recommendations.put(userID, items);
    }
    return recommendations;
  }
}
