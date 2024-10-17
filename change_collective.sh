#!/bin/bash

if [ $1 == 'leonardo' ]; then
    RULE_UPDATER_EXEC=/leonardo/home/userexternal/spasqual/Swing_Test/update_collective_rules
    RULE_FILE_PATH=/leonardo/home/userexternal/spasqual/Swing_Test/collective_rules.txt
elif [ $1 == 'local' ]; then
    RULE_UPDATER_EXEC=./update_collective_rules
    RULE_FILE_PATH=/home/saverio/University/Tesi/test/collective_rules.txt
else
    echo "ERROR: location not correctly set up, aborting..."
    exit 1
fi


$RULE_UPDATER_EXEC $RULE_FILE_PATH $2

export OMPI_MCA_coll_tuned_dynamic_rules_filename=${RULE_FILE_PATH}
