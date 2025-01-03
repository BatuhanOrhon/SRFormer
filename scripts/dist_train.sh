CONFIG=$1

if [ $# -lt 1 ] ;then
    echo "usage:"
    echo "./scripts/train.sh [path to option file]"
    exit
fi

PYTHONPATH="$(dirname $0)/..:${PYTHONPATH}" \
python basicsr/train.py -opt $CONFIG --launcher none