#!/bin/bash

rm ~/.ssh/known_hosts
scp ddboline@ddbolineathome.mooo.com:~/setup_files/build/kaggle_titantic_tutorial/*.csv .

./mymodel.py

ssh ddboline@ddbolineathome.mooo.com "~/bin/send_to_gtalk DONE_TITANTIC"
