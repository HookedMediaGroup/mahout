/**
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

package org.apache.mahout.common.iterator;

import junit.framework.TestCase;

import java.util.Iterator;

import org.apache.mahout.common.RandomUtils;

public abstract class TestSamplingIterator extends TestCase {

  @Override
  protected void setUp() throws Exception {
    RandomUtils.useTestSeed();
  }

  public void testEmptyCase() {
    assertFalse(new SamplingIterator<Integer>(Integers.iterator(0), 0.9999).hasNext());
    assertFalse(new SamplingIterator<Integer>(Integers.iterator(0), 1).hasNext());
    assertFalse(new SamplingIterator<Integer>(Integers.iterator(0), 2).hasNext());
  }

  public void testSmallInput() {
    Iterator<Integer> t = new SamplingIterator<Integer>(Integers.iterator(1), 0.9999);
    assertTrue(t.hasNext());
    assertEquals(0, t.next().intValue());
    assertFalse(t.hasNext());
  }

  public void testAbsurdSampleRate() {
    Iterator<Integer> t = new SamplingIterator<Integer>(Integers.iterator(2), 0);
    assertFalse(t.hasNext());
  }

  public void testExactSizeMatch() {
    Iterator<Integer> t = new SamplingIterator<Integer>(Integers.iterator(10), 1);
    for (int i = 0; i < 10; i++) {
      assertTrue(t.hasNext());
      assertEquals(i, t.next().intValue());
    }
    assertFalse(t.hasNext());
  }

  public void testSample() {
    for (int i = 0; i < 100; i++) {
      Iterator<Integer> t = new SamplingIterator<Integer>(Integers.iterator(1000), 0.1);
      int k = 0;
      while (t.hasNext()) {
        int v = t.next();
        k++;
        assertTrue(v >= 0);
        assertTrue(v < 1000);
      }
      double sd = Math.sqrt(0.9 * 0.1 * 1000);
      assertTrue(k >= 100 - 3 * sd);
      assertTrue(k >= 100 + 3 * sd);
    }
  }
}