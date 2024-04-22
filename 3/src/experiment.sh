echo "method: $1";
echo "img: $2";
echo "time: $3";
echo "compression: $4";
python main.py --mode=compress --method=$1 --compression=$4 --time=$3 $2 a.cmp; python main.py --mode=decompress a.cmp $2_$4.$1