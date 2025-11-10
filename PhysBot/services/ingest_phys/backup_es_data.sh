#!/bin/bash

CONTAINER_NAME="elasticsearch"
VOLUME_NAME="esdata_physbot"
BACKUP_DIR="./es_backup"

echo "📁 Creating local backup directory: $BACKUP_DIR"
mkdir -p $BACKUP_DIR

echo "📦 Copying data from container..."
docker cp ${CONTAINER_NAME}:/usr/share/elasticsearch/data $BACKUP_DIR

echo "✅ Done! Elasticsearch data backed up to: $BACKUP_DIR/data"