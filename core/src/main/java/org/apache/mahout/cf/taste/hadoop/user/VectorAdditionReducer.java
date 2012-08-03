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

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.cf.taste.hadoop.TasteHadoopUtils;
import org.apache.mahout.cf.taste.hadoop.item.PrefAndSimilarityColumnWritable;
import org.apache.mahout.math.VarLongWritable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

/**
 * sums the preference similarity columns together and outputs item vectors,
 * with user preferences per item
 */
public final class VectorAdditionReducer
    extends
    Reducer<VarLongWritable, PrefAndSimilarityColumnWritable, IntWritable, VectorWritable> {

  private boolean booleanData;

  @Override
  protected void setup(Context context) throws IOException {
    Configuration conf = context.getConfiguration();

    booleanData = conf.getBoolean(RecommenderJob.BOOLEAN_DATA, false);
  }

  @Override
  protected void reduce(VarLongWritable userID,
      Iterable<PrefAndSimilarityColumnWritable> values, Context context)
      throws IOException, InterruptedException {
    Vector v;

    if (booleanData) {
      v = VectorAdditionUtils.reduceBooleanData(userID, values);
    } else {
      v = VectorAdditionUtils.reduceNonBooleanData(userID, values);
    }

    if (v != null) {
      context.write(new IntWritable(TasteHadoopUtils.idToIndex(userID.get())),
          new VectorWritable(v));
    }
  }

}
