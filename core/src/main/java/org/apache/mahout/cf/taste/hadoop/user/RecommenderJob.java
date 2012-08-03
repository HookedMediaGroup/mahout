/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.cf.taste.hadoop.user;

import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.cf.taste.hadoop.RecommendedItemsWritable;
import org.apache.mahout.cf.taste.hadoop.item.PartialMultiplyMapper;
import org.apache.mahout.cf.taste.hadoop.item.PrefAndSimilarityColumnWritable;
import org.apache.mahout.cf.taste.hadoop.item.SimilarityMatrixRowWrapperMapper;
import org.apache.mahout.cf.taste.hadoop.item.ToVectorAndPrefReducer;
import org.apache.mahout.cf.taste.hadoop.item.UserVectorSplitterMapper;
import org.apache.mahout.cf.taste.hadoop.item.VectorAndPrefsWritable;
import org.apache.mahout.cf.taste.hadoop.item.VectorOrPrefWritable;
import org.apache.mahout.cf.taste.hadoop.preparation.PreparePreferenceMatrixJob;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.mapreduce.TransposeMapper;
import org.apache.mahout.math.VarIntWritable;
import org.apache.mahout.math.VarLongWritable;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.similarity.cooccurrence.RowSimilarityJob;
import org.apache.mahout.math.hadoop.similarity.cooccurrence.measures.VectorSimilarityMeasures;

/**
 * <p>Runs a completely distributed user-based recommender job as a series of mapreduces.</p>
 * <p/>
 * <p>Preferences in the input file should look like {@code userID, itemID[, preferencevalue]}</p>
 * <p/>
 * <p>
 * Preference value is optional to accommodate applications that have no notion of a preference value (that is, the user
 * simply expresses a preference for an item, but no degree of preference).
 * </p>
 * <p/>
 * <p>
 * The preference value is assumed to be parseable as a {@code double}. The user IDs and item IDs are
 * parsed as {@code int}s or {@code long}s.
 * </p>
 * <p/>
 * <p>Command line arguments specific to this class are:</p>
 * <p/>
 * <ol>
 * <li>--input(path): Directory containing one or more text files with the preference data</li>
 * <li>--output(path): output path where recommender output should go</li>
 * <li>--similarityClassname (classname): Name of vector similarity class to instantiate or a predefined similarity
 * from {@link org.apache.mahout.math.hadoop.similarity.cooccurrence.measures.VectorSimilarityMeasure}</li>
 * <li>--itemsFile (path): only include item IDs from this file in the recommendations (optional)</li>
 * <li>--numRecommendations (integer): Number of recommendations to compute per user (10)</li>
 * <li>--booleanData (boolean): Treat input data as having no pref values (false)</li>
 * <li>--encodeLongsAsInts (boolean): Encode Long item values as ints (true)</li>
 * <li>--itemBased (boolean): Run the CF job as item-based, not user-based (true)</li>
 * <li>--maxSimilaritiesPerUser (integer): Maximum number of similarities considered per user (100)</li>
 * <li>--minPrefsPerUser (integer): ignore users with less preferences than this in the similarity computation (1)</li>
 * <li>--maxPrefsPerUserInUserSimilarity (integer): max number of preferences to consider per user in the user similarity computation phase,
 * users with more preferences will be sampled down (1000)</li>
 * <li>--threshold (double): discard user pairs with a similarity value below this</li>
 * </ol>
 * <p/>
 * <p>General command line options are documented in {@link AbstractJob}.</p>
 * <p/>
 * <p>Note that because of how Hadoop parses arguments, all "-D" arguments must appear before all other
 * arguments.</p>
 */
public final class RecommenderJob extends AbstractJob {

  public static final String BOOLEAN_DATA = "booleanData";

  private static final int DEFAULT_MAX_SIMILARITIES_PER_ITEM = 100;
  private static final int DEFAULT_MAX_PREFS_PER_USER = 1000;
  private static final int DEFAULT_MIN_PREFS_PER_USER = 1;

  @Override
  public int run(String[] args) throws Exception {
    
    addInputOption();
    addOutputOption();
    addOption("numRecommendations", "n", "Number of recommendations per user",
            String.valueOf(RecommendationsWriter.DEFAULT_NUM_RECOMMENDATIONS));
    addOption("itemsFile", null, "File of items to recommend for", null);
    addOption("booleanData", "b", "Treat input as without pref values", Boolean.FALSE.toString());
    addOption("encodeLongsAsInts", "elai", "Encode Long item values as ints (true)", Boolean.TRUE.toString());
    addOption("itemBased", "ib", "Run as item-based, not user-based (true)", Boolean.TRUE.toString());
    addOption("minPrefsPerUser", "mp", "ignore users with less preferences than this in the similarity computation "
            + "(default: " + DEFAULT_MIN_PREFS_PER_USER + ')', String.valueOf(DEFAULT_MIN_PREFS_PER_USER));
    addOption("maxSimilaritiesPerItem", "m", "Maximum number of similarities considered per item ",
            String.valueOf(DEFAULT_MAX_SIMILARITIES_PER_ITEM));
    addOption("maxPrefsPerUserInUserSimilarity", "mppuiis", "max number of preferences to consider per user in the " +
            "item similarity computation phase, users with more preferences will be sampled down (default: " +
            DEFAULT_MAX_PREFS_PER_USER + ')', String.valueOf(DEFAULT_MAX_PREFS_PER_USER));
    addOption("similarityClassname", "s", "Name of distributed similarity measures class to instantiate, " +
            "alternatively use one of the predefined similarities (" + VectorSimilarityMeasures.list() + ')', true);
    addOption("threshold", "tr", "discard item pairs with a similarity value below this", false);

    Map<String, List<String>> parsedArgs = parseArguments(args);
    if (parsedArgs == null) {
      return -1;
    }

    Path outputPath = getOutputPath();
    int numRecommendations = Integer.parseInt(getOption("numRecommendations"));
    String itemsFile = getOption("itemsFile");
    boolean booleanData = Boolean.valueOf(getOption("booleanData"));
    boolean encodeLongsAsInts = Boolean.valueOf(getOption("encodeLongsAsInts"));
    int minPrefsPerUser = Integer.parseInt(getOption("minPrefsPerUser"));
    int maxPrefsPerUserInUserSimilarity = Integer.parseInt(getOption("maxPrefsPerUserInUserSimilarity"));
    int maxSimilaritiesPerItem = Integer.parseInt(getOption("maxSimilaritiesPerItem"));
    String similarityClassname = getOption("similarityClassname");
    double threshold = hasOption("threshold") ?
            Double.parseDouble(getOption("threshold")) : RowSimilarityJob.NO_THRESHOLD;
    boolean itemBased = Boolean.valueOf(getOption("itemBased"));

    Path prepPath = getTempPath("preparePreferenceMatrix");
    Path similarityMatrixPath = getTempPath("similarityMatrix");
    Path prePartialMultiplyPath1 = getTempPath("prePartialMultiply1");
    Path prePartialMultiplyPath2 = getTempPath("prePartialMultiply2");
    Path partialMultiplyPath = getTempPath("partialMultiply");
    Path aggregationPath = getTempPath("aggregationPath");

    AtomicInteger currentPhase = new AtomicInteger();

    int numberOfUsers = -1;

    if (shouldRunNextPhase(parsedArgs, currentPhase)) {
      ToolRunner.run(getConf(), new PreparePreferenceMatrixJob(), new String[]{
              "--input", getInputPath().toString(),
              "--output", prepPath.toString(),
              "--maxPrefsPerUser", String.valueOf(maxPrefsPerUserInUserSimilarity),
              "--minPrefsPerUser", String.valueOf(minPrefsPerUser),
              "--booleanData", String.valueOf(booleanData),
              "--encodeLongsAsInts", String.valueOf(encodeLongsAsInts),
              "--tempDir", getTempPath().toString()});

      numberOfUsers = HadoopUtil.readInt(new Path(prepPath, PreparePreferenceMatrixJob.NUM_USERS), getConf());
    }

    if (shouldRunNextPhase(parsedArgs, currentPhase)) {

      /* special behavior if phase 1 is skipped */
      if (numberOfUsers == -1) {
        numberOfUsers = (int) HadoopUtil.countRecords(new Path(prepPath, PreparePreferenceMatrixJob.USER_VECTORS),
                PathType.LIST, null, getConf());
      }

      /* Once DistributedRowMatrix uses the hadoop 0.20 API, we should refactor this call to something like
       * new DistributedRowMatrix(...).rowSimilarity(...) */
      //calculate the co-occurrence matrix
      ToolRunner.run(getConf(), new RowSimilarityJob(), new String[]{
              "--input", new Path(prepPath, PreparePreferenceMatrixJob.RATING_MATRIX).toString(),
              "--output", similarityMatrixPath.toString(),
              "--numberOfColumns", String.valueOf(numberOfUsers),
              "--similarityClassname", similarityClassname,
              "--maxSimilaritiesPerRow", String.valueOf(maxSimilaritiesPerItem),
              "--excludeSelfSimilarity", String.valueOf(Boolean.TRUE),
              "--threshold", String.valueOf(threshold),
              "--outputColumns", String.valueOf(itemBased),
              "--tempDir", getTempPath().toString()});
    }

    //start the multiplication of the co-occurrence matrix by the user vectors
    if (shouldRunNextPhase(parsedArgs, currentPhase)) {
      Job prePartialMultiply1 = prepareJob(
              similarityMatrixPath, prePartialMultiplyPath1, SequenceFileInputFormat.class,
              SimilarityMatrixRowWrapperMapper.class, VarIntWritable.class, VectorOrPrefWritable.class,
              Reducer.class, VarIntWritable.class, VectorOrPrefWritable.class,
              SequenceFileOutputFormat.class);
      boolean succeeded = prePartialMultiply1.waitForCompletion(true);
      if (!succeeded) 
        return -1;
      //continue the multiplication
      Job prePartialMultiply2 = prepareJob(new Path(prepPath, PreparePreferenceMatrixJob.USER_VECTORS),
              prePartialMultiplyPath2, SequenceFileInputFormat.class, UserVectorSplitterMapper.class, VarIntWritable.class,
              VectorOrPrefWritable.class, Reducer.class, VarIntWritable.class, VectorOrPrefWritable.class,
              SequenceFileOutputFormat.class);
      prePartialMultiply2.getConfiguration().setInt(UserVectorSplitterMapper.MAX_PREFS_PER_USER_CONSIDERED,
          Integer.MAX_VALUE); // first version of user-based does not try to limit number of preferences
      prePartialMultiply2.getConfiguration().setBoolean(UserVectorSplitterMapper.ITEM_BASED, itemBased);
      succeeded = prePartialMultiply2.waitForCompletion(true);
      if (!succeeded) 
        return -1;
      //finish the job
      Job partialMultiply = prepareJob(
              new Path(prePartialMultiplyPath1 + "," + prePartialMultiplyPath2), partialMultiplyPath,
              SequenceFileInputFormat.class, Mapper.class, VarIntWritable.class, VectorOrPrefWritable.class,
              ToVectorAndPrefReducer.class, VarIntWritable.class, VectorAndPrefsWritable.class,
              SequenceFileOutputFormat.class);
      setS3SafeCombinedInputPath(partialMultiply, getTempPath(), prePartialMultiplyPath1, prePartialMultiplyPath2);
      succeeded = partialMultiply.waitForCompletion(true);
      if (!succeeded) 
        return -1;
    }

    if (shouldRunNextPhase(parsedArgs, currentPhase)) {
      String aggregateAndRecommendInput = partialMultiplyPath.toString();
      //extract out the recommendations
      Job aggregate = prepareJob(
              new Path(aggregateAndRecommendInput), aggregationPath, SequenceFileInputFormat.class,
              PartialMultiplyMapper.class, VarLongWritable.class, PrefAndSimilarityColumnWritable.class,
              VectorAdditionReducer.class, IntWritable.class, VectorWritable.class,
              SequenceFileOutputFormat.class);
      Configuration aggregateAndRecommendConf = aggregate.getConfiguration();
      setIOSort(aggregate);
      aggregateAndRecommendConf.setBoolean(BOOLEAN_DATA, booleanData);
      boolean succeeded = aggregate.waitForCompletion(true);
      if (!succeeded) 
        return -1;

      // Transpose matrix multiplication results and output them as recommendations
      Job recommend = prepareJob(
          aggregationPath, outputPath, SequenceFileInputFormat.class,
          TransposeMapper.class, IntWritable.class, VectorWritable.class,
          VectorToRecommendationsReducer.class, VarLongWritable.class, RecommendedItemsWritable.class,
          TextOutputFormat.class);
      Configuration recommendConf = recommend.getConfiguration();
      if (itemsFile != null) {
        recommendConf.set(RecommendationsWriter.ITEMS_FILE, itemsFile);
      }
      setIOSort(recommend);
      recommendConf.setInt(RecommendationsWriter.NUM_RECOMMENDATIONS, numRecommendations);
      recommendConf.setBoolean(BOOLEAN_DATA, booleanData);
      if (encodeLongsAsInts) {
        recommendConf.set(RecommendationsWriter.ITEMID_INDEX_PATH,
            new Path(prepPath, PreparePreferenceMatrixJob.ITEMID_INDEX).toString());
      }
      succeeded = recommend.waitForCompletion(true);
      if (!succeeded) {
        return -1;
      }
    }

    return 0;
  }

  private static void setIOSort(JobContext job) {
    Configuration conf = job.getConfiguration();
    conf.setInt("io.sort.factor", 100);
    String javaOpts = conf.get("mapred.map.child.java.opts"); // new arg name
    if (javaOpts == null) {
      javaOpts = conf.get("mapred.child.java.opts"); // old arg name
    }
    int assumedHeapSize = 512;
    if (javaOpts != null) {
      Matcher m = Pattern.compile("-Xmx([0-9]+)([mMgG])").matcher(javaOpts);
      if (m.find()) {
        assumedHeapSize = Integer.parseInt(m.group(1));
        String megabyteOrGigabyte = m.group(2);
        if ("g".equalsIgnoreCase(megabyteOrGigabyte)) {
          assumedHeapSize *= 1024;
        }
      }
    }
    // Cap this at 1024MB now; see https://issues.apache.org/jira/browse/MAPREDUCE-2308
    conf.setInt("io.sort.mb", Math.min(assumedHeapSize / 2, 1024));
    // For some reason the Merger doesn't report status for a long time; increase
    // timeout when running these jobs
    conf.setInt("mapred.task.timeout", 60 * 60 * 1000);
  }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new RecommenderJob(), args);
  }
}
