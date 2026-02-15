
DATA_DIR=.
TEXTFILE=data.txt

for year in `ls $DATA_DIR`
do
    if [ -d "${DATA_DIR}/${year}" ]; then
        for month in `ls ${DATA_DIR}/${year}`
        do
            if [ -d "${DATA_DIR}/${year}/${month}" ]; then
                for datafile in `ls ${DATA_DIR}/${year}/${month}`
                do
                    echo ${DATA_DIR}/${year}/${month}/$datafile >> $TEXTFILE
                done
            fi
        done
    fi
done
