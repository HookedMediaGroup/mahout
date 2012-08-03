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

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.cf.taste.common.TopK;
import org.apache.mahout.cf.taste.hadoop.RecommendedItemsWritable;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.math.VarLongWritable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

/**
 * Converts vectors into a sorted and capped list of recommendations
 */
public final class VectorToRecommendationsReducer extends
    Reducer<IntWritable,VectorWritable,VarLongWritable,RecommendedItemsWritable> {

  private RecommendationsWriter recWriteUtils;

  @Override
  protected void setup(Context context) throws IOException {
    recWriteUtils = new RecommendationsWriter(context.getConfiguration());
  }

  @Override
  protected void reduce(IntWritable userID,
                        Iterable<VectorWritable> values,
                        Context context) throws IOException, InterruptedException {
    VectorWritable vectorWritable = VectorWritable.merge(values.iterator());
    Vector vector = vectorWritable.get();

    TopK<RecommendedItem> topKItems = recWriteUtils.writeRecommendedItems(new VarLongWritable(userID.get()), vector);

    if (!topKItems.isEmpty()) {
      context.write(new VarLongWritable(userID.get()), new RecommendedItemsWritable(topKItems.retrieve()));
    }
  }

}
