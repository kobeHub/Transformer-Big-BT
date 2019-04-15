#!/bin/bash
path=$1
if [ ! -f "$path" ]; then
  echo Create new script $path
  touch $path
else
  echo The file is already exists.
fi

echo '"""' >> $path
echo ' Author: Inno Jia @ https://kobehub.github.io' >> $path
echo ' Date:'   `date` >> $path
echo ' ' >> $path
echo '"""' >> $path
