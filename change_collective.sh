#!/bin/bash

# location='local'
# location='leonardo'
location='snellius'

if [ $location == 'leonardo' ]; then
    RULE_UPDATER_EXEC=$HOME/Swing_Test/update_collective_rules
    RULE_FILE_PATH=$HOME/Swing_Test/collective_rules.txt
elif [ $location == 'snellius' ]; then
    RULE_UPDATER_EXEC=$HOME/Swing_Test/update_collective_rules
    RULE_FILE_PATH=$HOME/Swing_Test/collective_rules.txt
elif [ $location == 'local' ]; then
    RULE_UPDATER_EXEC=./update_collective_rules
    RULE_FILE_PATH=$HOME/University/Tesi/test/collective_rules.txt
else
    echo "ERROR: location not correctly set up, aborting..."
    exit 1
fi


$RULE_UPDATER_EXEC $RULE_FILE_PATH $1

export OMPI_MCA_coll_tuned_use_dynamic_rules=1
export OMPI_MCA_coll_tuned_dynamic_rules_filename=${RULE_FILE_PATH}
