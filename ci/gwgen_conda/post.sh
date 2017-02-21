#!/bin/bash

# script that is called after the installation of gwgen_conda to ask whether
# an alias for gwgen shall be created

if [[ $BATCH == 0 ]] # interactive mode
then
    BASH_RC=$HOME/'<<<BASH_RC>>>'
    DEFAULT=yes
    echo -n "Do you wish to create an alias for gwgen in your $BASH_RC ? [yes|no]
[$DEFAULT] >>> "
    read ans
    if [[ $ans == "" ]]; then
        ans=$DEFAULT
    fi
    if [[ ($ans != "yes") && ($ans != "Yes") && ($ans != "YES") &&
                ($ans != "y") && ($ans != "Y") ]]
    then
        echo "
You may wish to edit your $BASH_RC:

$ alias gwgen=$PREFIX/bin/gwgen
"
    else
        if [ -f $BASH_RC ]; then
            echo "
Creating alias for gwgen in $BASH_RC
A backup will be made to: ${BASH_RC}-gwgen_conda_alias.bak
"
            cp $BASH_RC ${BASH_RC}-gwgen_conda_alias.bak
        else
            echo "
Creating alias for gwgen in $BASH_RC in
newly created $BASH_RC"
        fi
        echo "
For this change to become active, you have to open a new terminal.
"
        echo "
# added by gwgen_conda installer
alias gwgen=\"$PREFIX/bin/gwgen\"" >>$BASH_RC
    fi
fi
