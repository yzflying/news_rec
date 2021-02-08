#!/usr/bin/env bash

export JAVA_HOME=/usr/java/jdk1.8.0_181
export HADOOP_HOME=/opt/cloudera/parcels/CDH-6.3.2-1.cdh6.3.2.p0.1605554/lib/hadoop
export PATH=$PATH:$JAVA_HOME/bin:$HADOOP_HOME/bin

/opt/cloudera/parcels/CDH-6.3.2-1.cdh6.3.2.p0.1605554/lib/flume-ng/bin/flume-ng agent -c /opt/cloudera/parcels/CDH-6.3.2-1.cdh6.3.2.p0.1605554/lib/flume-ng/conf -f /opt/cloudera/parcels/CDH-6.3.2-1.cdh6.3.2.p0.1605554/lib/flume-ng/conf/collect_click.conf -Dflume.root.logger=INFO,console -name a1